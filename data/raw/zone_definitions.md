# Meridian Logistics — Delivery Zone Definitions (Synthetic)

> **Disclaimer:** This is a synthetic document created for educational purposes.
> All data, names, and figures are fictitious and do not represent real-world logistics operations.

---

## Overview

Meridian Logistics organises last-mile delivery into geographic zones. Each zone is
assigned to a set of postal codes and governs routing rules, fleet allocation,
SLA commitments, and cost structures.

---

## Zone Classification

### Zone 1 — Urban Core
- **Coverage:** Central city districts, typically postcodes with population
  density above 5,000 inhabitants per km².
- **Fleet type:** Electric cargo bikes, compact vans (max 3.5t), walking couriers.
- **Max stops per route:** 80 stops/day for vans, 40 stops/day for cargo bikes.
- **Standard SLA:** Next-day delivery by 12:00 for Priority shipments;
  by 18:00 for Standard.
- **Vendor fleet allowed:** Yes, up to 30% of daily volume.
- **Example cities:** Porto Centro, London Zone 1, Paris 1er–8e arrondissements.

### Zone 2 — Urban Periphery
- **Coverage:** Suburban residential areas within 20 km of city core.
- **Fleet type:** Medium vans (up to 7.5t), some electric vehicles.
- **Max stops per route:** 100 stops/day.
- **Standard SLA:** Next-day delivery by 18:00 for all service types.
- **Vendor fleet allowed:** Yes, up to 50% of daily volume.
- **Example cities:** Porto — Matosinhos, Maia; London — Zone 3–4 boroughs.

### Zone 3 — Semi-Rural
- **Coverage:** Towns and villages between 20–60 km from the nearest hub.
- **Fleet type:** Large vans (up to 12t), occasional HGV trunking.
- **Max stops per route:** 60 stops/day.
- **Standard SLA:** Next-day delivery by 18:00 where volume justifies daily
  frequency; otherwise 2-day delivery.
- **Vendor fleet allowed:** Yes, up to 70% of daily volume.

### Zone 4 — Rural / Remote
- **Coverage:** Postcodes more than 60 km from the nearest delivery hub,
  low-density areas.
- **Fleet type:** Large vans and HGVs only.
- **Max stops per route:** 35 stops/day.
- **Standard SLA:** 2-day delivery standard; next-day available as premium
  surcharge service only.
- **Vendor fleet allowed:** Yes, up to 90% of daily volume. Owned
  fleet may not be commercially viable.

---

## Zone Reclassification Policy

Zones are reviewed quarterly. A zone may be reclassified if any of the
following thresholds are met for two consecutive quarters:

| Trigger | Reclassification direction |
|---|---|
| Average daily volume increases >25% | One zone lower (e.g. Z3 → Z2) |
| Average daily volume decreases >20% | One zone higher (e.g. Z2 → Z3) |
| New hub opens within 30 km | Eligible for Zone 1 or 2 review |
| Customer density drops below 200/km² | Eligible for Zone 4 review |

---

## Zone-to-Hub Assignment

Each zone is anchored to a primary hub and may have a secondary hub for
overflow or contingency routing.

| Zone | Primary Hub | Secondary Hub | Max trunking distance |
|---|---|---|---|
| Zone 1 | Porto Urban Hub | N/A | 15 km |
| Zone 2 | Porto Urban Hub | Maia Satellite | 30 km |
| Zone 3 | Aveiro Regional Hub | Porto Urban Hub | 65 km |
| Zone 4 | Coimbra Hub | Aveiro Regional Hub | 130 km |
