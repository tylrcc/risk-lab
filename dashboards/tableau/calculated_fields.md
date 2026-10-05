# Tableau calculated fields

Create each one on the `scored_loans` data source unless noted.

**Default Rate**
```
SUM([actual_default]) / COUNT([loan_id])
```

**Approval Rate**
```
SUM(IF [decision] = "approve" THEN 1 ELSE 0 END) / COUNT([loan_id])
```

**Default Rate Approved**
```
SUM(IF [decision] = "approve" THEN [actual_default] END)
/ SUM(IF [decision] = "approve" THEN 1 END)
```

**Recall**
```
SUM(IF [decision] = "decline" AND [actual_default] = 1 THEN 1 END)
/ SUM([actual_default])
```

**Precision**
```
SUM(IF [decision] = "decline" AND [actual_default] = 1 THEN 1 END)
/ SUM(IF [decision] = "decline" THEN 1 END)
```

**Calibration Gap**
```
AVG([pd]) - [Default Rate]
```

**Outcome** (for colour legends)
```
IF [actual_default] = 1 THEN "Defaulted" ELSE "Repaid" END
```

**PD Bin** (or right click `pd`, Create, Bins, size 0.025)
```
FLOOR([pd] / 0.025) * 0.025
```

On the `returns` data source:

**Annualised Vol**
```
WINDOW_STDEV(SUM([return])) * SQRT(252)
```

**Max Drawdown**
```
WINDOW_MIN(MIN([drawdown]))
```
