#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include <atomic>

using namespace duckdb;

//===--------------------------------------------------------------------===//
// A mock "remote" scan: named in-memory row sets that can absorb joins with
// other scans of the same function (TableFunction::pushdown_join), the way a
// remote-database scan would ship the join to its server.
//===--------------------------------------------------------------------===//

static std::atomic<int> mock_fold_count {0};

struct MockRemoteScan {
	struct SourceTable {
		vector<string> names;
		vector<LogicalType> types;
		vector<vector<Value>> rows;
	};

	static const SourceTable &GetSource(const string &name) {
		static const case_insensitive_map_t<SourceTable> sources = []() {
			case_insensitive_map_t<SourceTable> result;
			result["r1"] = SourceTable {{"id", "v"},
			                            {LogicalType::INTEGER, LogicalType::INTEGER},
			                            {{Value::INTEGER(1), Value::INTEGER(100)},
			                             {Value::INTEGER(2), Value::INTEGER(200)},
			                             {Value::INTEGER(4), Value::INTEGER(400)}}};
			result["r2"] = SourceTable {{"id", "w"},
			                            {LogicalType::INTEGER, LogicalType::INTEGER},
			                            {{Value::INTEGER(1), Value::INTEGER(111)},
			                             {Value::INTEGER(3), Value::INTEGER(333)},
			                             {Value::INTEGER(4), Value::INTEGER(444)}}};
			result["r3"] = SourceTable {{"id", "u"},
			                            {LogicalType::INTEGER, LogicalType::INTEGER},
			                            {{Value::INTEGER(1), Value::INTEGER(1000)}, {Value::INTEGER(4), Value::INTEGER(4000)}}};
			return result;
		}();
		auto entry = sources.find(name);
		if (entry == sources.end()) {
			throw InvalidInputException("unknown mock source \"%s\"", name);
		}
		return entry->second;
	}

	struct MockBindData : public TableFunctionData {
		string server;
		vector<string> col_names;
		vector<LogicalType> col_types;
		vector<vector<Value>> rows;

		unique_ptr<FunctionData> Copy() const override {
			auto result = make_uniq<MockBindData>();
			result->server = server;
			result->col_names = col_names;
			result->col_types = col_types;
			result->rows = rows;
			return std::move(result);
		}
		bool Equals(const FunctionData &other) const override {
			return false;
		}
	};

	struct MockGlobalState : public GlobalTableFunctionState {
		idx_t row = 0;
	};

	static unique_ptr<FunctionData> Bind(ClientContext &context, TableFunctionBindInput &input,
	                                     vector<LogicalType> &return_types, vector<Identifier> &names) {
		auto &source = GetSource(input.inputs[0].ToString());
		auto result = make_uniq<MockBindData>();
		result->server = "srv";
		result->col_names = source.names;
		result->col_types = source.types;
		result->rows = source.rows;
		return_types = source.types;
		for (auto &name : source.names) {
			names.emplace_back(name);
		}
		return std::move(result);
	}

	static unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &context, TableFunctionInitInput &input) {
		return make_uniq<MockGlobalState>();
	}

	static void Function(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
		auto &bind_data = data.bind_data->Cast<MockBindData>();
		auto &state = data.global_state->Cast<MockGlobalState>();
		idx_t count = 0;
		while (state.row < bind_data.rows.size() && count < STANDARD_VECTOR_SIZE) {
			auto &row = bind_data.rows[state.row];
			for (idx_t col = 0; col < row.size(); col++) {
				output.SetValue(col, count, row[col]);
			}
			state.row++;
			count++;
		}
		output.SetCardinality(count);
	}

	//! Resolve a condition side to a column position in the given scan's bind data, or to a constant
	static bool ResolveSide(const Expression &expr, const LogicalGet &get, optional_idx &column, Value &constant) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			constant = expr.Cast<BoundConstantExpression>().GetValue();
			return true;
		}
		if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		auto &binding = expr.Cast<BoundColumnRefExpression>().Binding();
		if (binding.table_index != get.table_index) {
			return false;
		}
		auto &column_ids = get.GetColumnIds();
		auto slot = binding.column_index.GetIndex();
		if (slot >= column_ids.size()) {
			return false;
		}
		column = column_ids[slot].GetPrimaryIndex();
		return true;
	}

	static bool PushdownJoin(ClientContext &context, LogicalGet &left_get, LogicalGet &right_get,
	                         FunctionData *left_bind_data, FunctionData *right_bind_data, JoinType join_type,
	                         const vector<JoinCondition> &conditions, vector<LogicalType> &new_types,
	                         vector<Identifier> &new_names) {
		if (!left_bind_data || !right_bind_data) {
			return false;
		}
		auto &left = left_bind_data->Cast<MockBindData>();
		auto &right = right_bind_data->Cast<MockBindData>();
		if (left.server != right.server) {
			return false;
		}
		if (join_type != JoinType::INNER && join_type != JoinType::LEFT) {
			return false;
		}
		// resolve the conditions against both sides before mutating anything
		vector<pair<idx_t, idx_t>> pairs;
		bool never_match = false;
		for (auto &condition : conditions) {
			if (condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
				return false;
			}
			optional_idx left_col;
			optional_idx right_col;
			Value left_const;
			Value right_const;
			if (!ResolveSide(condition.GetLHS(), left_get, left_col, left_const) ||
			    !ResolveSide(condition.GetRHS(), right_get, right_col, right_const)) {
				return false;
			}
			if (left_col.IsValid() && right_col.IsValid()) {
				pairs.emplace_back(left_col.GetIndex(), right_col.GetIndex());
			} else if (!left_col.IsValid() && !right_col.IsValid()) {
				// constant = constant: either trivially true or a join that can never match
				if (left_const.IsNull() || right_const.IsNull() ||
				    ValueOperations::DistinctFrom(left_const, right_const)) {
					never_match = true;
				}
			} else {
				// constant against a column is not part of this mock's repertoire
				return false;
			}
		}

		vector<vector<Value>> joined;
		for (auto &lrow : left.rows) {
			bool matched = false;
			if (!never_match) {
				for (auto &rrow : right.rows) {
					bool match = true;
					for (auto &pair : pairs) {
						auto &lval = lrow[pair.first];
						auto &rval = rrow[pair.second];
						if (lval.IsNull() || rval.IsNull() || ValueOperations::DistinctFrom(lval, rval)) {
							match = false;
							break;
						}
					}
					if (match) {
						matched = true;
						vector<Value> row = lrow;
						row.insert(row.end(), rrow.begin(), rrow.end());
						joined.push_back(std::move(row));
					}
				}
			}
			if (!matched && join_type == JoinType::LEFT) {
				vector<Value> row = lrow;
				for (auto &type : right.col_types) {
					row.push_back(Value(type));
				}
				joined.push_back(std::move(row));
			}
		}

		vector<string> merged_names = left.col_names;
		merged_names.insert(merged_names.end(), right.col_names.begin(), right.col_names.end());
		vector<LogicalType> merged_types = left.col_types;
		merged_types.insert(merged_types.end(), right.col_types.begin(), right.col_types.end());

		new_types = merged_types;
		for (auto &name : merged_names) {
			new_names.emplace_back(name);
		}
		left.col_names = std::move(merged_names);
		left.col_types = std::move(merged_types);
		left.rows = std::move(joined);
		mock_fold_count++;
		return true;
	}

	static void Register(Connection &con) {
		con.BeginTransaction();
		auto &client_context = *con.context;
		auto &catalog = Catalog::GetSystemCatalog(client_context);
		TableFunction mock_remote_scan("mock_remote_scan", {LogicalType::VARCHAR}, MockRemoteScan::Function,
		                               MockRemoteScan::Bind, MockRemoteScan::InitGlobal);
		mock_remote_scan.pushdown_join = MockRemoteScan::PushdownJoin;
		CreateTableFunctionInfo info(mock_remote_scan);
		catalog.CreateTableFunction(*con.context, info);
		con.Commit();
	}
};

static void SetupUnionViews(Connection &con) {
	REQUIRE_NO_FAIL(con.Query("create table t1(id int, v int)"));
	REQUIRE_NO_FAIL(con.Query("insert into t1 values (1, 10), (2, 20), (5, 50)"));
	REQUIRE_NO_FAIL(con.Query("create table t2(id int, w int)"));
	REQUIRE_NO_FAIL(con.Query("insert into t2 values (1, 11), (5, 55), (6, 66)"));
	REQUIRE_NO_FAIL(con.Query("create table t3(id int, u int)"));
	REQUIRE_NO_FAIL(con.Query("insert into t3 values (5, 500), (6, 600)"));
	REQUIRE_NO_FAIL(con.Query("create view u1 as (select 'a' as site, * from t1) "
	                          "union all (select 'b' as site, * from mock_remote_scan('r1'))"));
	REQUIRE_NO_FAIL(con.Query("create view u2 as (select 'a' as site, * from t2) "
	                          "union all (select 'b' as site, * from mock_remote_scan('r2'))"));
	REQUIRE_NO_FAIL(con.Query("create view u3 as (select 'a' as site, * from t3) "
	                          "union all (select 'b' as site, * from mock_remote_scan('r3'))"));
}

//! EXPLAIN titles a scan box with a humanised operator name and repeats the raw function
//! name in its "Function:" row, so one scan box contributes one match for the raw name.
static idx_t CountOccurrences(const string &haystack, const string &needle) {
	idx_t count = 0;
	for (auto pos = haystack.find(needle); pos != string::npos; pos = haystack.find(needle, pos + needle.size())) {
		count++;
	}
	return count;
}

//! The physical plan rendered by EXPLAIN for the given query
static string ExplainPlan(Connection &con, const string &query) {
	auto result = con.Query("explain " + query);
	REQUIRE_NO_FAIL(*result);
	string plan;
	for (idx_t row = 0; row < result->RowCount(); row++) {
		plan += result->Collection().GetValue(1, row).ToString();
	}
	return plan;
}

TEST_CASE("Join over tag-disjoint UNION ALLs decomposes and folds into the scan", "[optimizer]") {
	DuckDB db(nullptr);
	Connection con(db);
	MockRemoteScan::Register(con);
	SetupUnionViews(con);
	mock_fold_count = 0;

	const string inner_join = "select p.site, p.id, p.v, q.w from u1 p "
	                          "join u2 q on q.site = p.site and q.id = p.id order by p.site, p.id";

	// correct results with the rewrite active...
	auto result = con.Query(inner_join);
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(*result, 0, {"a", "a", "b", "b"}));
	REQUIRE(CHECK_COLUMN(*result, 1, {1, 5, 1, 4}));
	REQUIRE(CHECK_COLUMN(*result, 2, {10, 50, 100, 400}));
	REQUIRE(CHECK_COLUMN(*result, 3, {11, 55, 111, 444}));

	// ...the same-server pair actually folded...
	REQUIRE(mock_fold_count > 0);

	// ...and the folded plan reads a single merged mock scan with no join above it
	auto plan = ExplainPlan(con, inner_join);
	REQUIRE(CountOccurrences(plan, "MOCK_REMOTE_SCAN") == 1); // one merged scan

	// identical results with the pass disabled
	REQUIRE_NO_FAIL(con.Query("set disabled_optimizers = 'join_union_pushdown'"));
	auto vanilla = con.Query(inner_join);
	REQUIRE_NO_FAIL(*vanilla);
	REQUIRE(CHECK_COLUMN(*vanilla, 0, {"a", "a", "b", "b"}));
	REQUIRE(CHECK_COLUMN(*vanilla, 1, {1, 5, 1, 4}));
	REQUIRE(CHECK_COLUMN(*vanilla, 2, {10, 50, 100, 400}));
	REQUIRE(CHECK_COLUMN(*vanilla, 3, {11, 55, 111, 444}));
	auto vanilla_plan = ExplainPlan(con, inner_join);
	REQUIRE(CountOccurrences(vanilla_plan, "MOCK_REMOTE_SCAN") == 2); // two separate scans
}

TEST_CASE("LEFT joins over tag-disjoint UNION ALLs NULL-pad within their leg", "[optimizer]") {
	DuckDB db(nullptr);
	Connection con(db);
	MockRemoteScan::Register(con);
	SetupUnionViews(con);
	mock_fold_count = 0;

	const string left_join = "select p.site, p.id, p.v, q.w from u1 p "
	                         "left join u2 q on q.site = p.site and q.id = p.id order by p.site, p.id";

	auto result = con.Query(left_join);
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(*result, 0, {"a", "a", "a", "b", "b", "b"}));
	REQUIRE(CHECK_COLUMN(*result, 1, {1, 2, 5, 1, 2, 4}));
	REQUIRE(CHECK_COLUMN(*result, 2, {10, 20, 50, 100, 200, 400}));
	REQUIRE(CHECK_COLUMN(*result, 3, {11, Value(), 55, 111, Value(), 444}));
	REQUIRE(mock_fold_count > 0);
	REQUIRE(CountOccurrences(ExplainPlan(con, left_join), "MOCK_REMOTE_SCAN") == 1); // one merged scan
}

TEST_CASE("Three-view joins decompose iteratively into one merged scan per site", "[optimizer]") {
	DuckDB db(nullptr);
	Connection con(db);
	MockRemoteScan::Register(con);
	SetupUnionViews(con);
	mock_fold_count = 0;

	const string triple_join = "select p.id, r.u from u1 p "
	                           "join u2 q on q.site = p.site and q.id = p.id "
	                           "join u3 r on r.site = p.site and r.id = p.id order by p.id";

	auto result = con.Query(triple_join);
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(*result, 0, {1, 4, 5}));
	REQUIRE(CHECK_COLUMN(*result, 1, {1000, 4000, 500}));
	// both same-server joins folded, leaving a single merged mock scan
	REQUIRE(mock_fold_count >= 2);
	REQUIRE(CountOccurrences(ExplainPlan(con, triple_join), "MOCK_REMOTE_SCAN") == 1); // one merged scan
}

TEST_CASE("Joins without tag equality do not decompose", "[optimizer]") {
	DuckDB db(nullptr);
	Connection con(db);
	MockRemoteScan::Register(con);
	SetupUnionViews(con);
	mock_fold_count = 0;

	// sites may mix: cross-leg matches are real matches
	auto result = con.Query("select count(*) from u1 p join u2 q on q.id = p.id");
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(*result, 0, {6}));
	REQUIRE(mock_fold_count == 0);
}

TEST_CASE("UNION (distinct) legs do not decompose", "[optimizer]") {
	DuckDB db(nullptr);
	Connection con(db);
	MockRemoteScan::Register(con);
	SetupUnionViews(con);
	mock_fold_count = 0;

	REQUIRE_NO_FAIL(con.Query("create view u1d as (select 'a' as site, * from t1) "
	                          "union (select 'b' as site, * from mock_remote_scan('r1'))"));
	auto result = con.Query("select count(*) from u1d p join u2 q on q.site = p.site and q.id = p.id");
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(*result, 0, {4}));
	REQUIRE(mock_fold_count == 0);
}
