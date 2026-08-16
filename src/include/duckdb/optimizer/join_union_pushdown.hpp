//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_union_pushdown.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace duckdb {
class Optimizer;

//! JoinUnionPushdown decomposes a join between two UNION ALLs whose legs are paired by a distinct
//! constant "leg tag" (a per-leg literal equated by a join condition, so cross-leg pairs are
//! provably empty) into a union of per-leg joins, and collapses a per-leg join between two scans
//! of the same table function into a single scan via TableFunction::pushdown_join. The
//! decomposition only commits when at least one per-leg join is absorbed by its scan.
class JoinUnionPushdown : public LogicalOperatorVisitor {
public:
	explicit JoinUnionPushdown(Optimizer &optimizer);

	void VisitOperator(unique_ptr<LogicalOperator> &op) override;

	//! Resolve a set operation leg's output column to its defining expression, descending through
	//! column-reference projections, filters and (inner/left) comparison joins; returns the
	//! defining expression, or nullptr when it cannot be traced.
	static const Expression *ResolveLegColumn(const LogicalOperator &leg, idx_t column);

private:
	bool TryDecomposeJoinOverUnions(unique_ptr<LogicalOperator> &op);
	//! Collapse a join of two same-function scans into the left scan; replaces op with a
	//! projection over the merged scan. register_replacements redirects the surrounding plan's
	//! bindings (union legs are positional and need no redirection).
	bool TryCollapseJoin(unique_ptr<LogicalOperator> &op, bool register_replacements);
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override;

	static bool ExprReferencesOnly(const Expression &expr, TableIndex table_index);
	static bool ExprColumnBinding(const Expression &expr, ColumnBinding &binding);
	static void RemapExprToLeg(unique_ptr<Expression> &expr, TableIndex setop_index,
	                           const vector<ColumnBinding> &leg_bindings, const vector<LogicalType> &leg_types);
	static bool InlineProjectionInto(unique_ptr<Expression> &expr, const LogicalProjection &proj);
	static bool BuildChainOutputExpressions(LogicalOperator &chain_top, const LogicalGet &get,
	                                        vector<unique_ptr<Expression>> &result);

private:
	Optimizer &optimizer;
	column_binding_map_t<ColumnBinding> replacement_map;
};
} // namespace duckdb
