#include "duckdb/optimizer/join_union_pushdown.hpp"

#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"

namespace duckdb {

JoinUnionPushdown::JoinUnionPushdown(Optimizer &optimizer_p) : optimizer(optimizer_p) {
}

//===--------------------------------------------------------------------===//
// Joins over UNION ALL
//===--------------------------------------------------------------------===//
// A join between two UNION ALLs whose legs each carry a distinct constant "leg tag" that a join
// condition equates (e.g. region-partitioned union views joined on the region column) can never
// match rows across differently-tagged legs, so it is equivalent to a union of per-leg joins.
// Decomposing pays off when a per-leg join sits between two scans of the same table function
// that can absorb it (TableFunction::pushdown_join) - e.g. two scans of one remote server, which
// then executes the join and returns only its result. The decomposition is speculative: it only
// commits when at least one per-leg join is absorbed.

static bool IsDecomposableJoinType(JoinType type) {
	return type == JoinType::INNER || type == JoinType::LEFT;
}

bool JoinUnionPushdown::ExprReferencesOnly(const Expression &expr, TableIndex table_index) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		return expr.Cast<BoundColumnRefExpression>().Binding().table_index == table_index;
	}
	bool result = true;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!ExprReferencesOnly(child, table_index)) {
			result = false;
		}
	});
	return result;
}

bool JoinUnionPushdown::ExprColumnBinding(const Expression &expr, ColumnBinding &binding) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	binding = expr.Cast<BoundColumnRefExpression>().Binding();
	return true;
}

//! Whether every table referenced by the expression is the given one (an expression without
//! column references qualifies for any table)
static bool WithinTable(const Expression &expr, TableIndex table_index) {
	unordered_set<TableIndex> bindings;
	LogicalJoin::GetExpressionBindings(expr, bindings);
	for (auto &binding : bindings) {
		if (binding != table_index) {
			return false;
		}
	}
	return true;
}

const Expression *JoinUnionPushdown::ResolveLegColumn(const LogicalOperator &leg, idx_t column) {
	auto op = &leg;
	idx_t col = column;
	while (true) {
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_PROJECTION: {
			if (col >= op->expressions.size()) {
				return nullptr;
			}
			auto &expr = *op->expressions[col];
			if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return &expr;
			}
			auto &binding = expr.Cast<BoundColumnRefExpression>().Binding();
			auto child_bindings = op->children[0]->GetColumnBindings();
			optional_idx next;
			for (idx_t i = 0; i < child_bindings.size(); i++) {
				if (child_bindings[i] == binding) {
					next = i;
					break;
				}
			}
			if (!next.IsValid()) {
				return nullptr;
			}
			col = next.GetIndex();
			op = op->children[0].get();
			break;
		}
		case LogicalOperatorType::LOGICAL_FILTER: {
			if (op->HasProjectionMap()) {
				return nullptr;
			}
			// a filter passes its child's columns through positionally
			op = op->children[0].get();
			break;
		}
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
			auto &join = op->Cast<LogicalComparisonJoin>();
			if (join.HasProjectionMap() || !IsDecomposableJoinType(join.join_type)) {
				return nullptr;
			}
			// the join emits its left child's columns followed by its right child's
			auto left_count = op->children[0]->GetColumnBindings().size();
			if (col < left_count) {
				op = op->children[0].get();
			} else if (join.join_type == JoinType::INNER) {
				col -= left_count;
				op = op->children[1].get();
			} else {
				// the NULL-padded side of an outer join does not preserve constants
				return nullptr;
			}
			break;
		}
		default:
			return nullptr;
		}
	}
}

//! The scan at the bottom of a projection-only chain, if it can absorb joins
static optional_ptr<LogicalGet> FindJoinPushdownGet(LogicalOperator &leg) {
	reference<LogicalOperator> op(leg);
	while (op.get().type == LogicalOperatorType::LOGICAL_PROJECTION && op.get().children.size() == 1) {
		op = *op.get().children[0];
	}
	if (op.get().type != LogicalOperatorType::LOGICAL_GET) {
		return nullptr;
	}
	auto &get = op.get().Cast<LogicalGet>();
	if (!get.function.pushdown_join) {
		return nullptr;
	}
	if (!get.children.empty() || !get.projected_input.empty()) {
		return nullptr;
	}
	return &get;
}

//! Remap column references from the union's bindings to the leg's own output
void JoinUnionPushdown::RemapExprToLeg(unique_ptr<Expression> &expr, TableIndex setop_index,
                                       const vector<ColumnBinding> &leg_bindings,
                                       const vector<LogicalType> &leg_types) {
	ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
	    expr, [&](BoundColumnRefExpression &col_ref, unique_ptr<Expression> &node) {
		    if (col_ref.Binding().table_index != setop_index) {
			    return;
		    }
		    auto col = col_ref.Binding().column_index.GetIndex();
		    node = make_uniq<BoundColumnRefExpression>(col_ref.GetAlias(), leg_types[col], leg_bindings[col]);
	    });
}

//! Substitute column references to a projection by (copies of) the projection's expressions
bool JoinUnionPushdown::InlineProjectionInto(unique_ptr<Expression> &expr, const LogicalProjection &proj) {
	bool ok = true;
	ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
	    expr, [&](BoundColumnRefExpression &col_ref, unique_ptr<Expression> &node) {
		    if (col_ref.Binding().table_index != proj.table_index) {
			    return;
		    }
		    auto col = col_ref.Binding().column_index.GetIndex();
		    if (col >= proj.expressions.size()) {
			    ok = false;
			    return;
		    }
		    node = proj.expressions[col]->Copy();
	    });
	return ok;
}

//! The chain top's output expressions with every projection inlined, so all column references
//! point at the scan at the bottom of the chain; false when a reference cannot be traced.
bool JoinUnionPushdown::BuildChainOutputExpressions(LogicalOperator &chain_top, const LogicalGet &get,
                                                    vector<unique_ptr<Expression>> &result) {
	if (chain_top.type == LogicalOperatorType::LOGICAL_GET) {
		auto bindings = chain_top.GetColumnBindings();
		chain_top.ResolveOperatorTypes();
		for (idx_t i = 0; i < bindings.size(); i++) {
			result.push_back(make_uniq<BoundColumnRefExpression>(chain_top.types[i], bindings[i]));
		}
		return true;
	}
	auto &top = chain_top.Cast<LogicalProjection>();
	for (auto &expr : top.expressions) {
		result.push_back(expr->Copy());
	}
	reference<LogicalOperator> node(*chain_top.children[0]);
	while (node.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = node.get().Cast<LogicalProjection>();
		for (auto &expr : result) {
			if (!JoinUnionPushdown::InlineProjectionInto(expr, proj)) {
				return false;
			}
		}
		node = *node.get().children[0];
	}
	for (auto &expr : result) {
		if (!JoinUnionPushdown::ExprReferencesOnly(*expr, get.table_index)) {
			return false;
		}
	}
	return true;
}

bool JoinUnionPushdown::TryCollapseJoin(unique_ptr<LogicalOperator> &op, bool register_replacements) {
	if (op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return false;
	}
	auto &join = op->Cast<LogicalComparisonJoin>();
	if (!IsDecomposableJoinType(join.join_type) || join.HasProjectionMap() ||
	    !join.duplicate_eliminated_columns.empty() || join.conditions.empty()) {
		return false;
	}
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison()) {
			return false;
		}
	}
	auto left_get = FindJoinPushdownGet(*join.children[0]);
	auto right_get = FindJoinPushdownGet(*join.children[1]);
	if (!left_get || !right_get || left_get.get() == right_get.get()) {
		return false;
	}
	if (left_get->function.name != right_get->function.name ||
	    left_get->function.pushdown_join != right_get->function.pushdown_join) {
		return false;
	}
	auto &context = optimizer.context;

	join.children[0]->ResolveOperatorTypes();
	join.children[1]->ResolveOperatorTypes();
	const auto old_left_top = join.children[0]->GetColumnBindings();
	const auto old_right_top = join.children[1]->GetColumnBindings();

	// copy the conditions, orienting each so its left expression belongs to the left child
	vector<TableIndex> side_index;
	for (idx_t side = 0; side < 2; side++) {
		auto &child = *join.children[side];
		side_index.push_back(child.type == LogicalOperatorType::LOGICAL_PROJECTION
		                         ? child.Cast<LogicalProjection>().table_index
		                         : child.Cast<LogicalGet>().table_index);
	}
	vector<JoinCondition> conditions;
	for (auto &condition : join.conditions) {
		auto copy = condition.Copy();
		bool straight = WithinTable(copy.GetLHS(), side_index[0]) && WithinTable(copy.GetRHS(), side_index[1]);
		bool swapped = WithinTable(copy.GetLHS(), side_index[1]) && WithinTable(copy.GetRHS(), side_index[0]);
		if (straight) {
			// keep as-is
		} else if (swapped) {
			copy.Swap();
		} else {
			return false;
		}
		conditions.push_back(std::move(copy));
	}

	// inline both projection chains into the conditions so every reference points at its scan
	for (idx_t side = 0; side < 2; side++) {
		reference<LogicalOperator> node(*join.children[side]);
		while (node.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = node.get().Cast<LogicalProjection>();
			for (auto &condition : conditions) {
				if (!JoinUnionPushdown::InlineProjectionInto(side == 0 ? condition.LeftReference() : condition.RightReference(), proj)) {
					return false;
				}
			}
			node = *node.get().children[0];
		}
	}
	for (auto &condition : conditions) {
		if (!JoinUnionPushdown::ExprReferencesOnly(condition.GetLHS(), left_get->table_index) ||
		    !JoinUnionPushdown::ExprReferencesOnly(condition.GetRHS(), right_get->table_index)) {
			return false;
		}
	}

	// the replacement projection's expressions: both children's outputs, traced down to the scans
	vector<unique_ptr<Expression>> left_exprs;
	vector<unique_ptr<Expression>> right_exprs;
	if (!BuildChainOutputExpressions(*join.children[0], *left_get, left_exprs) ||
	    !BuildChainOutputExpressions(*join.children[1], *right_get, right_exprs)) {
		return false;
	}

	// the merged scan must produce the left scan's output columns followed by the right's
	left_get->ResolveOperatorTypes();
	right_get->ResolveOperatorTypes();
	const auto left_get_bindings = left_get->GetColumnBindings();
	const auto right_get_bindings = right_get->GetColumnBindings();
	auto expected_types = left_get->types;
	expected_types.insert(expected_types.end(), right_get->types.begin(), right_get->types.end());

	vector<LogicalType> new_types;
	vector<Identifier> new_names;
	if (!left_get->function.pushdown_join(context, *left_get, *right_get, left_get->bind_data.get(),
	                                      right_get->bind_data.get(), join.join_type, conditions, new_types,
	                                      new_names)) {
		DUCKDB_LOG_INFO(context, "join_union_pushdown: scan \"%s\" declined to absorb the join",
		                left_get->function.name.GetIdentifierName());
		return false;
	}
	if (new_types != expected_types || new_names.size() != new_types.size()) {
		throw InternalException("pushdown_join returned a schema that does not match the two scans' output columns");
	}

	// re-point the left scan at the merged result
	auto new_index = optimizer.binder.GenerateTableIndex();
	left_get->table_index = new_index;
	left_get->returned_types = new_types;
	left_get->names = std::move(new_names);
	vector<ColumnIndex> column_ids;
	for (idx_t i = 0; i < new_types.size(); i++) {
		column_ids.emplace_back(i);
	}
	left_get->SetColumnIds(std::move(column_ids));
	left_get->projection_ids.clear();
	left_get->table_filters = TableFilterSet();
	left_get->ResolveOperatorTypes();
	left_get->estimated_cardinality = join.estimated_cardinality;
	left_get->has_estimated_cardinality = join.has_estimated_cardinality;

	// the projection reads the merged scan: the left scan's slots first, then the right's
	column_binding_map_t<ColumnBinding> slot_map;
	for (idx_t i = 0; i < left_get_bindings.size(); i++) {
		slot_map[left_get_bindings[i]] = ColumnBinding(new_index, ProjectionIndex(i));
	}
	for (idx_t i = 0; i < right_get_bindings.size(); i++) {
		slot_map[right_get_bindings[i]] = ColumnBinding(new_index, ProjectionIndex(left_get_bindings.size() + i));
	}
	auto remap_onto_merged = [&](vector<unique_ptr<Expression>> &exprs) {
		for (auto &expr : exprs) {
			ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
			    expr, [&](BoundColumnRefExpression &col_ref, unique_ptr<Expression> &node) {
				    auto entry = slot_map.find(col_ref.Binding());
				    if (entry != slot_map.end()) {
					    col_ref.BindingMutable() = entry->second;
				    }
			    });
		}
	};
	remap_onto_merged(left_exprs);
	remap_onto_merged(right_exprs);

	// extract the merged scan from the bottom of the left chain
	reference<unique_ptr<LogicalOperator>> holder(join.children[0]);
	while (holder.get()->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		holder = holder.get()->children[0];
	}
	auto merged_scan = std::move(holder.get());

	auto projection_exprs = std::move(left_exprs);
	for (auto &expr : right_exprs) {
		projection_exprs.push_back(std::move(expr));
	}
	auto projection_index = optimizer.binder.GenerateTableIndex();
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projection_exprs));
	projection->children.push_back(std::move(merged_scan));
	projection->ResolveOperatorTypes();
	projection->estimated_cardinality = join.estimated_cardinality;
	projection->has_estimated_cardinality = join.has_estimated_cardinality;

	if (register_replacements) {
		for (idx_t i = 0; i < old_left_top.size(); i++) {
			replacement_map[old_left_top[i]] = ColumnBinding(projection_index, ProjectionIndex(i));
		}
		for (idx_t i = 0; i < old_right_top.size(); i++) {
			replacement_map[old_right_top[i]] =
			    ColumnBinding(projection_index, ProjectionIndex(old_left_top.size() + i));
		}
	}
	op = std::move(projection);
	return true;
}

bool JoinUnionPushdown::TryDecomposeJoinOverUnions(unique_ptr<LogicalOperator> &op) {
	if (op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return false;
	}
	auto &join = op->Cast<LogicalComparisonJoin>();
	if (!IsDecomposableJoinType(join.join_type) || join.HasProjectionMap() ||
	    !join.duplicate_eliminated_columns.empty() || join.conditions.empty()) {
		return false;
	}
	// generated join filters are only attached by the (later) JOIN_FILTER_PUSHDOWN pass
	D_ASSERT(!join.filter_pushdown);
	if (join.children[0]->type != LogicalOperatorType::LOGICAL_UNION ||
	    join.children[1]->type != LogicalOperatorType::LOGICAL_UNION) {
		return false;
	}
	auto &left_union = join.children[0]->Cast<LogicalSetOperation>();
	auto &right_union = join.children[1]->Cast<LogicalSetOperation>();
	if (!left_union.setop_all || !right_union.setop_all) {
		return false;
	}
	auto leg_count = left_union.children.size();
	if (leg_count < 2 || right_union.children.size() != leg_count) {
		return false;
	}
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison()) {
			return false;
		}
	}
	// only worthwhile when both unions contain a scan that could absorb a per-leg join
	bool any_candidate = false;
	for (auto &left_leg : left_union.children) {
		auto left_get = FindJoinPushdownGet(*left_leg);
		if (!left_get) {
			continue;
		}
		for (auto &right_leg : right_union.children) {
			auto right_get = FindJoinPushdownGet(*right_leg);
			if (right_get && left_get->function.name == right_get->function.name) {
				any_candidate = true;
				break;
			}
		}
		if (any_candidate) {
			break;
		}
	}
	if (!any_candidate) {
		return false;
	}
	auto &context = optimizer.context;

	// normalize condition orientation: left expression from the left union, right from the right
	vector<JoinCondition> conditions;
	for (auto &condition : join.conditions) {
		auto copy = condition.Copy();
		bool straight =
		    WithinTable(copy.GetLHS(), left_union.table_index) && WithinTable(copy.GetRHS(), right_union.table_index);
		bool swapped =
		    WithinTable(copy.GetLHS(), right_union.table_index) && WithinTable(copy.GetRHS(), left_union.table_index);
		if (straight) {
			// keep as-is
		} else if (swapped) {
			copy.Swap();
		} else {
			DUCKDB_LOG_INFO(context, "join_union_pushdown: not decomposed, a join condition straddles the unions");
			return false;
		}
		conditions.push_back(std::move(copy));
	}

	// find a tag condition: an equality between columns that are non-NULL constants, pairwise
	// distinct within each union, with equal tag sets across the two - pairing the legs
	vector<idx_t> pairing;
	for (auto &condition : conditions) {
		if (condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
			continue;
		}
		ColumnBinding left_binding;
		ColumnBinding right_binding;
		if (!JoinUnionPushdown::ExprColumnBinding(condition.GetLHS(), left_binding) ||
		    !JoinUnionPushdown::ExprColumnBinding(condition.GetRHS(), right_binding)) {
			continue;
		}
		auto resolve_tags = [](const LogicalSetOperation &setop, idx_t column, vector<Value> &tags) {
			for (auto &leg : setop.children) {
				auto expr = ResolveLegColumn(*leg, column);
				if (!expr || expr->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
					return false;
				}
				auto &value = expr->Cast<BoundConstantExpression>().GetValue();
				if (value.IsNull()) {
					return false;
				}
				tags.push_back(value);
			}
			for (idx_t i = 0; i < tags.size(); i++) {
				for (idx_t j = i + 1; j < tags.size(); j++) {
					if (!ValueOperations::DistinctFrom(tags[i], tags[j])) {
						return false;
					}
				}
			}
			return true;
		};
		vector<Value> left_tags;
		vector<Value> right_tags;
		if (!resolve_tags(left_union, left_binding.column_index.GetIndex(), left_tags) ||
		    !resolve_tags(right_union, right_binding.column_index.GetIndex(), right_tags)) {
			continue;
		}
		// pair each left leg with the right leg carrying the same tag
		vector<idx_t> candidate;
		for (auto &left_tag : left_tags) {
			optional_idx match;
			for (idx_t j = 0; j < right_tags.size(); j++) {
				if (!ValueOperations::DistinctFrom(left_tag, right_tags[j])) {
					match = j;
					break;
				}
			}
			if (!match.IsValid()) {
				break;
			}
			candidate.push_back(match.GetIndex());
		}
		if (candidate.size() == leg_count) {
			pairing = std::move(candidate);
			break;
		}
	}
	if (pairing.empty()) {
		DUCKDB_LOG_INFO(context,
		                "join_union_pushdown: not decomposed, no distinct constant leg tag pairs the union legs");
		return false;
	}

	// build the per-leg joins, absorbing each into its scan where possible
	vector<unique_ptr<LogicalOperator>> new_children;
	idx_t folded_count = 0;
	for (idx_t i = 0; i < leg_count; i++) {
		auto left_leg = std::move(left_union.children[i]);
		auto right_leg = std::move(right_union.children[pairing[i]]);
		left_leg->ResolveOperatorTypes();
		right_leg->ResolveOperatorTypes();
		auto left_bindings = left_leg->GetColumnBindings();
		auto right_bindings = right_leg->GetColumnBindings();
		auto &left_types = left_leg->types;
		auto &right_types = right_leg->types;
		auto leg_join = make_uniq<LogicalComparisonJoin>(join.join_type);
		for (auto &condition : conditions) {
			auto copy = condition.Copy();
			JoinUnionPushdown::RemapExprToLeg(copy.LeftReference(), left_union.table_index, left_bindings, left_types);
			JoinUnionPushdown::RemapExprToLeg(copy.RightReference(), right_union.table_index, right_bindings,
			                                  right_types);
			leg_join->conditions.push_back(std::move(copy));
		}
		leg_join->children.push_back(std::move(left_leg));
		leg_join->children.push_back(std::move(right_leg));
		leg_join->ResolveOperatorTypes();
		leg_join->estimated_cardinality = join.estimated_cardinality;
		leg_join->has_estimated_cardinality = join.has_estimated_cardinality;
		unique_ptr<LogicalOperator> leg_op = std::move(leg_join);
		if (TryCollapseJoin(leg_op, false)) {
			folded_count++;
		}
		new_children.push_back(std::move(leg_op));
	}
	if (folded_count == 0) {
		// no scan absorbed its per-leg join - decomposing alone has no benefit; restore the legs
		for (idx_t i = 0; i < leg_count; i++) {
			auto &leg_join = *new_children[i];
			left_union.children[i] = std::move(leg_join.children[0]);
			right_union.children[pairing[i]] = std::move(leg_join.children[1]);
		}
		DUCKDB_LOG_INFO(context, "join_union_pushdown: not decomposed, no scan absorbed its per-leg join");
		return false;
	}

	auto left_count = left_union.column_count;
	auto right_count = right_union.column_count;
	auto new_setop = make_uniq<LogicalSetOperation>(optimizer.binder.GenerateTableIndex(), left_count + right_count,
	                                                std::move(new_children), LogicalOperatorType::LOGICAL_UNION,
	                                                /*setop_all=*/true,
	                                                left_union.allow_out_of_order && right_union.allow_out_of_order);
	new_setop->ResolveOperatorTypes();
	new_setop->estimated_cardinality = join.estimated_cardinality;
	new_setop->has_estimated_cardinality = join.has_estimated_cardinality;

	for (idx_t i = 0; i < left_count; i++) {
		replacement_map[ColumnBinding(left_union.table_index, ProjectionIndex(i))] =
		    ColumnBinding(new_setop->table_index, ProjectionIndex(i));
	}
	for (idx_t i = 0; i < right_count; i++) {
		replacement_map[ColumnBinding(right_union.table_index, ProjectionIndex(i))] =
		    ColumnBinding(new_setop->table_index, ProjectionIndex(left_count + i));
	}
	op = std::move(new_setop);
	return true;
}

void JoinUnionPushdown::VisitOperator(unique_ptr<LogicalOperator> &op) {
	LogicalOperatorVisitor::VisitOperator(op);
	if (!TryDecomposeJoinOverUnions(op)) {
		TryCollapseJoin(op, /*register_replacements=*/true);
	}
}

unique_ptr<Expression> JoinUnionPushdown::VisitReplace(BoundColumnRefExpression &expr,
                                                       unique_ptr<Expression> *expr_ptr) {
	// follow the chain: a rewrite's output union may itself be consumed by a later rewrite
	// before the expressions referencing it are visited
	auto entry = replacement_map.find(expr.Binding());
	while (entry != replacement_map.end()) {
		expr.BindingMutable() = entry->second;
		entry = replacement_map.find(expr.Binding());
	}
	return nullptr;
}

} // namespace duckdb
