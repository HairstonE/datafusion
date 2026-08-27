// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`GroupValues`] trait for storing and interning group keys

use arrow::array::types::{
    Date32Type, Date64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

use arrow::array::{ArrayRef, downcast_primitive};
use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion_common::{Result, ScalarValue};

use datafusion_expr::EmitTo;

pub mod multi_group_by;

mod row;
pub use row::GroupValuesRows;
mod single_group_by;
use datafusion_physical_expr::binary_map::OutputType;
use datafusion_physical_expr::expressions::Column;
use multi_group_by::GroupValuesColumn;

use crate::statistics::{StatisticsArgs, StatisticsContext};

pub(crate) use single_group_by::primitive::HashValue;

use crate::aggregates::{
    AggregateExec,
    group_values::single_group_by::{
        boolean::GroupValuesBoolean, bytes::GroupValuesBytes,
        bytes_view::GroupValuesBytesView, flat::GroupValuesFlatPrimitive,
        primitive::GroupValuesPrimitive,
    },
    order::GroupOrdering,
};

mod metrics;
mod null_builder;

pub(crate) use metrics::GroupByMetrics;

/// Stores the group values during hash aggregation.
///
/// # Background
///
/// In a query such as `SELECT a, b, count(*) FROM t GROUP BY a, b`, the group values
/// identify each group, and correspond to all the distinct values of `(a,b)`.
///
/// ```sql
/// -- Input has 4 rows with 3 distinct combinations of (a,b) ("groups")
/// create table t(a int, b varchar)
/// as values (1, 'a'), (2, 'b'), (1, 'a'), (3, 'c');
///
/// select a, b, count(*) from t group by a, b;
/// ----
/// 1 a 2
/// 2 b 1
/// 3 c 1
/// ```
///
/// # Design
///
/// Managing group values is a performance critical operation in hash
/// aggregation. The major operations are:
///
/// 1. Intern: Quickly finding existing and adding new group values
/// 2. Emit: Returning the group values as an array
///
/// There are multiple specialized implementations of this trait optimized for
/// different data types and number of columns, optimized for these operations.
/// See [`new_group_values`] for details.
///
/// # Group Ids
///
/// Each distinct group in a hash aggregation is identified by a unique group id
/// (usize) which is assigned by instances of this trait. Group ids are
/// continuous without gaps, starting from 0.
pub trait GroupValues: Send {
    /// Calculates the group id for each input row of `cols`, assigning new
    /// group ids as necessary.
    ///
    /// When the function returns, `groups`  must contain the group id for each
    /// row in `cols`.
    ///
    /// If a row has the same value as a previous row, the same group id is
    /// assigned. If a row has a new value, the next available group id is
    /// assigned.
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()>;

    /// Returns the number of bytes of memory used by this [`GroupValues`].
    ///
    /// May be expensive; check the implementation before calling on hot paths.
    fn size(&self) -> usize;

    /// Returns true if this [`GroupValues`] is empty
    fn is_empty(&self) -> bool;

    /// The number of values (distinct group values) stored in this [`GroupValues`]
    fn len(&self) -> usize;

    /// Emits the group values
    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>>;

    /// Clear the contents and shrink the capacity to the size of the batch (free up memory usage)
    fn clear_shrink(&mut self, num_rows: usize);
}

pub(crate) struct FlatStatsHint {
    min: ScalarValue,
    max: ScalarValue,
    distinct: Option<usize>,
}

pub fn new_group_values(
    schema: SchemaRef,
    group_ordering: &GroupOrdering,
) -> Result<Box<dyn GroupValues>> {
    new_group_values_hinted(schema, group_ordering, None)
}

/// Return a specialized implementation of [`GroupValues`] for the given schema.
///
/// [`GroupValues`] implementations choosing logic:
///
///   - If group by single column, and type of this column has
///     the specific [`GroupValues`] implementation, such implementation
///     will be chosen.
///
///   - If group by multiple columns, and all column types have the specific
///     `GroupColumn` implementations, `GroupValuesColumn` will be chosen.
///
///   - Otherwise, the general implementation `GroupValuesRows` will be chosen.
///
/// `GroupColumn`:  crate::aggregates::group_values::multi_group_by::GroupColumn
/// `GroupValuesColumn`: crate::aggregates::group_values::multi_group_by::GroupValuesColumn
/// `GroupValuesRows`: crate::aggregates::group_values::GroupValuesRows
pub(crate) fn new_group_values_hinted(
    schema: SchemaRef,
    group_ordering: &GroupOrdering,
    hint: Option<FlatStatsHint>,
) -> Result<Box<dyn GroupValues>> {
    if schema.fields.len() == 1 {
        let d = schema.fields[0].data_type();

        macro_rules! flat_helper {
            ($t:ty) => {
                return Ok(Box::new(GroupValuesFlatPrimitive::<$t>::with_hint(
                    d.clone(),
                    hint,
                )))
            };
        }
        match d {
            DataType::Int8 => flat_helper!(Int8Type),
            DataType::Int16 => flat_helper!(Int16Type),
            DataType::Int32 => flat_helper!(Int32Type),
            DataType::Int64 => flat_helper!(Int64Type),
            DataType::UInt8 => flat_helper!(UInt8Type),
            DataType::UInt16 => flat_helper!(UInt16Type),
            DataType::UInt32 => flat_helper!(UInt32Type),
            DataType::UInt64 => flat_helper!(UInt64Type),
            DataType::Date32 => flat_helper!(Date32Type),
            DataType::Date64 => flat_helper!(Date64Type),
            DataType::Time32(TimeUnit::Second) => flat_helper!(Time32SecondType),
            DataType::Time32(TimeUnit::Millisecond) => {
                flat_helper!(Time32MillisecondType)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                flat_helper!(Time64MicrosecondType)
            }
            DataType::Time64(TimeUnit::Nanosecond) => flat_helper!(Time64NanosecondType),
            DataType::Timestamp(TimeUnit::Second, _) => flat_helper!(TimestampSecondType),
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                flat_helper!(TimestampMillisecondType)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                flat_helper!(TimestampMicrosecondType)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                flat_helper!(TimestampNanosecondType)
            }
            _ => {}
        }

        macro_rules! downcast_helper {
            ($t:ty, $d:ident) => {
                return Ok(Box::new(GroupValuesPrimitive::<$t>::new($d.clone())))
            };
        }

        downcast_primitive! {
            d => (downcast_helper, d),
            _ => {}
        }

        match d {
            DataType::Decimal128(_, _) => {
                downcast_helper!(Decimal128Type, d);
            }
            DataType::Utf8 => {
                return Ok(Box::new(GroupValuesBytes::<i32>::new(OutputType::Utf8)));
            }
            DataType::LargeUtf8 => {
                return Ok(Box::new(GroupValuesBytes::<i64>::new(OutputType::Utf8)));
            }
            DataType::Utf8View => {
                return Ok(Box::new(GroupValuesBytesView::new(OutputType::Utf8View)));
            }
            DataType::Binary => {
                return Ok(Box::new(GroupValuesBytes::<i32>::new(OutputType::Binary)));
            }
            DataType::LargeBinary => {
                return Ok(Box::new(GroupValuesBytes::<i64>::new(OutputType::Binary)));
            }
            DataType::BinaryView => {
                return Ok(Box::new(GroupValuesBytesView::new(OutputType::BinaryView)));
            }
            DataType::Boolean => {
                return Ok(Box::new(GroupValuesBoolean::new()));
            }
            _ => {}
        }
    }

    if multi_group_by::supported_schema(schema.as_ref()) {
        if matches!(group_ordering, GroupOrdering::None) {
            Ok(Box::new(GroupValuesColumn::<false>::try_new(schema)?))
        } else {
            Ok(Box::new(GroupValuesColumn::<true>::try_new(schema)?))
        }
    } else {
        Ok(Box::new(GroupValuesRows::try_new(schema)?))
    }
}

pub(crate) fn flat_stats_hint(agg: &AggregateExec) -> Option<FlatStatsHint> {
    let [(expr, _)] = agg.group_by.expr() else {
        return None;
    };
    let col = expr.downcast_ref::<Column>()?;

    let schema = agg.input.schema();
    let field = schema.fields().get(col.index())?;
    if !field.data_type().is_integer() {
        return None;
    }
    let stats = StatisticsContext::new()
        .compute(agg.input.as_ref(), &StatisticsArgs::new())
        .ok()?;
    let cs = stats.column_statistics.get(col.index())?;
    Some(FlatStatsHint {
        min: cs.min_value.get_value()?.clone(),
        max: cs.max_value.get_value()?.clone(),
        distinct: cs.distinct_count.get_value().copied(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, TimestampNanosecondArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion_expr::EmitTo;
    use std::sync::Arc;

    fn single_col(dt: DataType) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("g", dt, true)]))
    }

    // A single temporal column must reach the flat grouper via dispatch and produce
    // correct group ids. A mis-routed arm (wrong `T`) would panic in `as_primitive::<T>`.
    #[test]
    fn dispatch_date32_groups_correctly() {
        let schema = single_col(DataType::Date32);
        let mut gv = new_group_values(schema, &GroupOrdering::None).unwrap();
        let col: ArrayRef =
            Arc::new(Date32Array::from(vec![Some(5), Some(5), None, Some(7)]));
        let mut groups = vec![];
        gv.intern(std::slice::from_ref(&col), &mut groups).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 2]);
        let out = gv.emit(EmitTo::All).unwrap();
        assert_eq!(out[0].data_type(), &DataType::Date32);
    }

    // End-to-end proof of the C4 fix through the public entry point: the emitted
    // array must keep the column's timezone, not the type constant's `None`.
    #[test]
    fn dispatch_timestamp_preserves_timezone() {
        let dt = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));
        let schema = single_col(dt.clone());
        let mut gv = new_group_values(schema, &GroupOrdering::None).unwrap();
        let col: ArrayRef = Arc::new(
            TimestampNanosecondArray::from(vec![Some(5i64), Some(5), None, Some(7)])
                .with_timezone("UTC"),
        );
        let mut groups = vec![];
        gv.intern(std::slice::from_ref(&col), &mut groups).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 2]);
        let out = gv.emit(EmitTo::All).unwrap();
        assert_eq!(out[0].data_type(), &dt);
    }
}
