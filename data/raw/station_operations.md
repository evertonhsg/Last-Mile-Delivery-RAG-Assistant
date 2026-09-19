# Meridian Logistics — Station Operations Manual (Synthetic)

> **Disclaimer:** Synthetic document for educational/portfolio purposes only.

---

## 1. Station Types

### Urban Hub Station
- **Role:** Primary sortation and dispatch point for Zone 1 and Zone 2 deliveries.
- **Operating hours:** 05:00–22:00 Monday–Saturday; 07:00–20:00 Sunday.
- **Inbound trunking window:** 04:00–06:30 (first wave), 10:00–11:30 (second wave).
- **Dispatch cut-off:** 08:30 (Priority routes), 10:00 (Standard routes).
- **Minimum staffing:** 4 sorters, 2 loaders, 1 supervisor per shift.

### Satellite Station
- **Role:** Last-mile dispatch for Zone 2 overflow and Zone 3 coverage.
- **Operating hours:** 06:00–20:00 Monday–Saturday.
- **Inbound trunking window:** 05:30–07:30 only (single wave).
- **Dispatch cut-off:** 09:00 (all route types).
- **Minimum staffing:** 2 sorters, 1 loader, 1 supervisor per shift.

### Pickup Point Partner
- **Role:** Customer collection point (PUDO — Pick Up Drop Off).
- **Operating hours:** Set by the partner (typically retail store hours).
- **Parcel hold period:** 7 calendar days before return to sender.
- **Max capacity:** 200 parcels per day (standard PUDO agreement).

---

## 2. Morning Sortation Process

The morning sortation runs as follows after inbound trunking:

1. **Unload & scan** — all inbound parcels scanned on conveyor belt (Meridian Scan Events entry: `UNLOAD`).
2. **Exception identification** — damaged, mislabelled, or hazmat parcels segregated (event: `EXCEPTION`).
3. **Route assignment** — system auto-assigns parcel to route based on postcode (event: `ROUTE_ASSIGN`).
4. **Loading** — parcels loaded into vehicles in reverse-stop-sequence order (last stop loaded first).
5. **Driver briefing** — 10-minute standup: route highlights, special instructions, weather alerts.
6. **Dispatch scan** — each vehicle scanned out of depot (event: `DISPATCH`).

Target: full sortation completed within 90 minutes of last inbound truck arrival.

---

## 3. End-of-Day Process

1. **Vehicle return** — all owned fleet must return to station by 21:00.
2. **POD reconciliation** — driver submits Proof of Delivery count; system reconciles
   against dispatched manifest (event: `RECONCILE`).
3. **Undelivered parcel handling:**
   - Parcel with remaining attempts → returned to sort cage for next day.
   - Parcel at max attempts → labelled for PUDO transfer or return-to-sender.
4. **Vehicle condition report** — driver completes digital checklist (damage, fuel/charge level).
5. **Station manager sign-off** — reviews SLA performance dashboard before 22:00.

---

## 4. Key Performance Indicators (Station Level)

| KPI | Definition | Daily target |
|---|---|---|
| Sortation accuracy | Parcels correctly routed / total sorted | ≥99.8% |
| On-time dispatch | Routes dispatched before cut-off / total routes | ≥99.0% |
| Delivery success rate | Successful first-attempt deliveries / total attempts | ≥93.0% |
| POD scan rate | Parcels with valid POD scan / total delivered | ≥99.5% |
| Undelivered rate | Undelivered at end of day / total dispatched | ≤5.0% |
| Customer complaint rate | Complaints per 1,000 deliveries | ≤2.5 |

---

## 5. Escalation Procedures

### Level 1 — Operational Issue (Station Manager handles)
- Single route SLA breach.
- Vehicle breakdown (spare vehicle available).
- Weather delay <2 hours.

### Level 2 — Tactical Issue (Regional Operations Manager notified)
- Station SLA below target for >3 consecutive days.
- Vehicle shortage affecting >20% of planned routes.
- Vendor performance below contractual SLA for >7 days.

### Level 3 — Strategic Issue (Division VP notified)
- Hub sortation failure (>50% of parcels unprocessed).
- Industrial action affecting station operations.
- SLA below target for >10 consecutive days at any hub station.

---

## 6. Health, Safety & Compliance

- Manual handling limit: 25 kg per parcel without mechanical assistance.
- All sorters complete manual handling training every 12 months.
- Hazmat parcels (ADR classified) must be handled by certified staff only.
- GDPR: Customer address labels must be shredded, not recycled, at end of day.
- Vehicle telematics data is retained for 90 days for compliance and route analysis.
