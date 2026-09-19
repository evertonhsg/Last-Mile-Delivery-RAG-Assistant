# Meridian Logistics — Routing Rules & Route Optimisation Standards (Synthetic)

> **Disclaimer:** Synthetic document for educational/portfolio purposes only.

---

## 1. Route Construction Principles

All last-mile routes at Meridian Logistics are constructed using the following
hierarchy of constraints:

1. **Hard constraints** (must never be violated):
   - Vehicle capacity (weight and volume limits).
   - Driver legal working hours: max 9h driving per day (EU Regulation 561/2006).
   - Time-window commitments for Priority Overnight stops.
   - Hazmat segregation rules (ADR-classified parcels cannot share with food).

2. **Soft constraints** (optimised, but can be relaxed under operational pressure):
   - Preferred stop sequence (minimise backtracking).
   - Fuel/energy efficiency (minimise total km driven).
   - Driver familiarity with area (returning drivers preferred for complex zones).

---

## 2. Route Types

### Static Routes
- Fixed sequence of stops reconstructed each day based on volume.
- Used in Zone 1 and Zone 2 where volume is high and predictable.
- Reviewed monthly by the Network Planning team.
- Advantage: simple to manage, easy to train new drivers.

### Dynamic Routes
- Constructed fresh each morning using the optimisation engine.
- Used in Zone 3 and Zone 4 where stop density varies daily.
- Inputs: parcel manifests, driver availability, vehicle capacity, traffic data.
- Advantage: more efficient, adapts to volume fluctuations.

### Mixed Routes
- Static backbone with dynamic overflow stops appended at end.
- Common in Zone 2 during peak periods (November–December, Easter).

---

## 3. Fleet Allocation Rules

Fleet is allocated in priority order based on route type and zone:

| Priority | Vehicle type | Preferred zone | Max load |
|---|---|---|---|
| 1 | Electric cargo bike | Zone 1 only | 100 kg / 0.5 m³ |
| 2 | Electric van (3.5t) | Zone 1, Zone 2 | 900 kg / 6 m³ |
| 3 | Diesel van (7.5t) | Zone 2, Zone 3 | 2,500 kg / 18 m³ |
| 4 | HGV (12t) | Zone 3, Zone 4 | 6,000 kg / 40 m³ |
| 5 | Vendor van | Any zone | Per vendor contract |

Vendor fleet is dispatched **only after** owned fleet is fully utilised for the day,
unless a vendor contract specifies guaranteed minimum volume.

---

## 4. Route Optimisation Algorithm

The Meridian Logistics planning system uses a **Vehicle Routing Problem with Time Windows
(VRPTW)** solver. The objective function minimises:

```
Total cost = α × distance + β × time + γ × SLA_penalty + δ × capacity_waste
```

Default weights: α=0.35, β=0.30, γ=0.25, δ=0.10.

During peak season (Nov 15 – Jan 6), γ is increased to 0.40 and δ reduced to
0.05 to prioritise SLA compliance over efficiency.

---

## 5. Dynamic Stop Addition (DSA)

Stops added after the route has been dispatched (typically from e-commerce
platforms with late cut-off times) follow the DSA protocol:

1. System checks if any active driver is within 2 km of the new stop.
2. If yes: stop is appended to that driver's manifest via mobile app.
3. If no: stop is held for next-day delivery unless Priority Overnight service.
4. Priority Overnight late additions are escalated to the station manager for
   manual decision.

---

## 6. Route Performance KPIs

| KPI | Definition | Target |
|---|---|---|
| Stop compliance rate | % of planned stops completed on day of dispatch | ≥98.5% |
| Route adherence | % of routes completed in planned sequence | ≥92.0% |
| On-road time variance | Actual vs planned drive time | ≤+15% |
| Km per stop | Total km driven / total stops completed | ≤1.8 km (Zone 1–2), ≤4.5 km (Zone 3–4) |
| Failed first attempt rate | Failed deliveries / total attempts | ≤7.0% |

KPIs are reported at route level, aggregated to station level daily.

---

## 7. Route Review Process

Routes are reviewed under these conditions:

- **Weekly:** Station manager reviews routes with adherence below 88%.
- **Monthly:** NP team reviews zone-level km-per-stop trends.
- **Triggered:** Any route generating >3 customer complaints in a 7-day period
  is flagged for immediate review.
- **Seasonal:** Full route restructure in October (pre-peak) and February
  (post-peak) each year.
