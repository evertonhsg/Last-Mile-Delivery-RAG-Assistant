# Meridian Logistics — Capacity Planning & Fleet Management Guidelines (Synthetic)

> **Disclaimer:** Synthetic document for educational/portfolio purposes only.

---

## 1. Capacity Planning Cycle

Meridian Logistics uses a rolling 13-week capacity planning cycle, updated every Monday.

### Planning Horizons
- **Week 1–2 (Operational):** Firm plan. Fleet and headcount locked.
  Changes require Station Manager approval.
- **Week 3–6 (Tactical):** Provisional plan. Volume forecasts drive fleet
  booking; vendor contracts activated or released here.
- **Week 7–13 (Strategic):** Indicative plan. Used for vendor negotiations,
  capital leasing decisions, and headcount hiring plans.

---

## 2. Demand Forecasting

Volume forecasts are produced by the Demand Planning team using an ensemble model:

| Model | Weight | Best for |
|---|---|---|
| XGBoost (gradient boosting) | 40% | Capturing non-linear volume spikes |
| Holt-Winters (exponential smoothing) | 35% | Seasonal baseline |
| ARIMA | 15% | Short-term trend correction |
| Human adjustment | 10% | Events, promotions, known anomalies |

Forecasts are generated at station × service-type × day granularity.
MAPE target: ≤8% at station level over a 7-day horizon.

---

## 3. Fleet Sizing Rules

Required fleet size is calculated daily using:

```
Required vans = ceil(Forecast volume / Target stops-per-van)
Buffer fleet  = ceil(Required vans × Buffer rate)
Total fleet   = Required vans + Buffer fleet
```

Buffer rates by zone:
- Zone 1: 8% buffer (low volatility, high density).
- Zone 2: 12% buffer.
- Zone 3: 18% buffer.
- Zone 4: 25% buffer (high volatility, route uncertainty).

---

## 4. Vendor Fleet Management

### Vendor Tier Classification

| Tier | Criteria | Benefits |
|---|---|---|
| Gold | ≥500 stops/day, ≥24-month relationship, SLA ≥97% | Guaranteed minimum volume commitment, priority dispatch |
| Silver | 200–499 stops/day, ≥12-month relationship, SLA ≥94% | Preferred booking, 30-day rolling contract |
| Bronze | <200 stops/day or <12-month relationship | Ad-hoc booking, no volume guarantee |

### Vendor SLA Requirements
- Vendors must achieve ≥93% on-time delivery rate measured over 28 days.
- Failure to meet SLA for 2 consecutive months triggers a performance review.
- Three consecutive months below SLA results in tier demotion or contract termination.

### Vendor Data Integration
- Gold and Silver vendors connect via the Meridian Vendor API for real-time tracking.
- Bronze vendors submit end-of-day scan files (CSV format, by 23:00 local time).

---

## 5. Peak Season Capacity Protocol

Peak season is defined as November 15 – January 6 each year.

### Pre-Peak Preparation (October)
1. Vendor fleet bookings increased by planned peak uplift percentage (typically 35–60%).
2. Temporary driver headcount contracts activated.
3. Overflow sortation capacity confirmed with hub operations.
4. Dynamic routing enabled for all zones.

### Peak Activation Triggers
Additional surge capacity is activated when:
- Daily volume exceeds forecast by >15% for 2 consecutive days.
- SLA performance drops below 95% at any station for 3 consecutive days.

### Peak De-escalation
Surge capacity stands down when:
- 7-day rolling volume drops to within 5% of non-peak forecast.
- Date passes January 7.

---

## 6. Capacity Utilisation Targets

| Metric | Target | Alert threshold |
|---|---|---|
| Fleet utilisation (owned) | 85–92% | <75% or >95% |
| Vendor volume share | ≤50% of daily volume | >60% for >3 days |
| Hub sortation utilisation | ≤88% of rated capacity | >92% |
| Driver overtime rate | ≤12% of driver-days | >20% |

When any metric crosses its alert threshold, an automated notification is
sent to the Station Manager and Regional Planning lead.
