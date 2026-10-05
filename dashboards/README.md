# Dashboards

Both pipelines write flat CSVs that load straight into Power BI or Tableau with no
transformation. Run the pipelines first, then point the tool at `reports/`.

```
reports/
  portfolio/
    returns.csv       date, series, return, cumulative, drawdown, rolling_vol_21d, rolling_var_95
    weights.csv       date, strategy, ticker, weight
    metrics.csv       one row per strategy and benchmark
    var_backtest.csv  Kupiec test results per strategy
  credit/
    scored_loans.csv  every held out loan with features, actual outcome, pd, rating, decision
    metrics.csv       one row per model
    feature_importance.csv
    rating_buckets.csv
    calibration.csv   predicted vs observed default rate by PD decile
    roc_curve.csv     fpr, tpr, threshold
```

## Power BI

1. Get data, Folder, select `reports/credit` (or `reports/portfolio`). Load each CSV as its own table.
2. Set `date` columns to Date type. Set `pd`, `return`, `weight` to Decimal.
3. Import `powerbi/measures.dax` through the DAX query view, or paste the measures one at a time into a new measure on the `scored_loans` table.
4. Apply `powerbi/theme.json` under View, Themes, Browse for themes.
5. Add a `rating` slicer and a `decision` slicer on every credit page.

Credit pages that work well:

| Page | Visuals |
|---|---|
| Overview | Cards: AUC, Gini, KS, Approval rate, Default rate (approved). Clustered column: observed default rate by `rating` with `avg_pd` as a line. |
| Model quality | Line chart `roc_curve` (tpr by fpr). Scatter `calibration` (observed by predicted) with a diagonal reference line. Bar `feature_importance` sorted descending. |
| Portfolio | Table `scored_loans` with conditional formatting on `pd`. Histogram of `pd` split by `actual_default`. |

Portfolio pages:

| Page | Visuals |
|---|---|
| Performance | Line `cumulative` by `date`, legend `series`. Area `drawdown` by `date`. Cards from `metrics` for the selected strategy. |
| Risk | Line `rolling_vol_21d` and `rolling_var_95`. Stacked area `weight` by `date`, legend `ticker`, filtered to one `strategy`. |

## Tableau

1. Connect to Text file, pick `scored_loans.csv`. Add the other CSVs as separate data sources (they share no keys, so leave them unrelated).
2. Paste the fields in `tableau/calculated_fields.md` into Analysis, Create Calculated Field.
3. Build the sheets listed below, then combine them on a dashboard with `rating` and `decision` as global filters.

| Sheet | Rows | Columns | Marks |
|---|---|---|---|
| Default rate by rating | AVG(actual_default), AVG(pd) | rating | Dual axis, bar and line |
| Score distribution | CNT(loan_id) | pd (bin size 0.025) | Bar, colour by actual_default |
| ROC | tpr | fpr | Line, sort by fpr |
| Calibration | observed | predicted | Circle, add a reference line y = x |
| Feature importance | feature (sorted by importance) | importance | Bar |
| Growth | cumulative | date | Line, colour by series |
| Allocation | weight | date | Stacked area, colour by ticker, filter strategy |
