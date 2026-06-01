# User Persona Pipeline — 180-Day Window Documentation

**Notebook:** `persona_pipeline copy 2.ipynb`
**SQL:** `persona_query.sql`
**Cache:** `persona_raw_180d.parquet`
**Method:** Two-Stage RFM Houses + Within-House K-means Clustering
**Data window:** Last 180 days (6 months)

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [How It Differs from the 90-Day Version](#2-how-it-differs-from-the-90-day-version)
3. [Input Data](#3-input-data)
4. [Feature Engineering](#4-feature-engineering)
5. [Stage 1 — RFM Scoring & House Assignment](#5-stage-1--rfm-scoring--house-assignment)
6. [Stage 2 — Within-House Clustering](#6-stage-2--within-house-clustering)
7. [What Each Persona Means](#7-what-each-persona-means)
8. [Evaluation](#8-evaluation)
9. [Known Issues & Limitations](#9-known-issues--limitations)
10. [Next Steps](#10-next-steps)

---

## 1. What This Pipeline Does

The pipeline assigns every user who visited in the last 6 months a **persona label** — a short description of *where they are in their lifecycle* and *how they consume content*.

It solves a core problem with standard K-means user segmentation: if you cluster all users together, engagement-level differences (who visits daily vs once a month) dominate the result and collapse 90%+ of users into one giant cluster. The pipeline avoids this with two stages:

**Stage 1 — RFM Houses (rule-based)**
Assigns each user to a lifecycle house based on how recently they visited (R), how often (F), and how deeply they engaged (M). This separates the lifecycle question before any ML runs.

**Stage 2 — Within-House Clustering (K-means)**
Within each house, users are already similar on engagement level. K-means then runs only on content preference, device, and time-of-day features — finding genuine behavioural sub-groups that are actionable for editorial and marketing.

**Output:** Labels like `Champions – Audio`, `At Risk - HV – News Heavy`, `Sleeping Giants – Culture`.

---

## 2. How It Differs from the 90-Day Version

| Dimension | 90-Day Pipeline | 180-Day Pipeline |
|---|---|---|
| **Data window** | 90 days | 180 days (6 months) |
| **Expected user volume** | ~1.35M | ~2–3M (approx 2× larger) |
| **R score bins** | 0–7, 8–21, 22–60, 61–90 | 0–7, 8–30, 31–90, 91–180 |
| **R=4 (lapsed) definition** | Last visit 61–90 days ago | Last visit 91–180 days ago |
| **Reactivation gap threshold** | 30 days | 60 days |
| **Reactivation return window** | Within last 30 days | Within last 60 days |
| **Cache file** | `persona_raw.parquet` | `persona_raw_180d.parquet` |
| **Dormant / Sleeping Giants** | Smaller (many lapsed users outside 90d) | Much larger — captures longer-lapsed users |
| **New user share** | ~25% (30d is large fraction of 90d) | ~10–15% (30d is smaller fraction of 180d) |

The 180-day window gives a much more complete picture of the user base, including users who visit monthly or less frequently. The 90-day window was biased towards very active users.

---

## 3. Input Data

### Source
- **Table:** `sdp_silver.studios_piano.studios_piano_events`
- **Event:** `page.display`
- **Platform:** `app_type IN ('responsive', 'web')`
- **Window:** Last 180 days
- **Exclusions:** keepalive, onboarding, error, notification pages; rows with null `user_id` or `content_pillar`

### Columns Returned

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `recency_days` | Days since last page view (within 180d window) |
| `account_age_days` | Days since first page view in the window (proxy for account age) |
| `total_visits_90d` | Distinct visit count over 180 days |
| `dwell_s_median` | Median dwell time per visit (seconds) |
| `pv_per_visit_avg` | Average page views per visit |
| `deep_read_rate` | Fraction of pages where dwell > 120 seconds |
| `reactivation_gap_days` | Size of gap (days) before a user returned after 60+ days absence; null if not applicable |
| `pv_morning/lunch/evening/night` | Raw page view counts by time-of-day slot |
| `pv_weekday/weekend` | Raw page view counts by day type |
| `frac_desktop` | Fraction of page views from desktop |
| `frac_pillar_*` | Fraction of page views per content pillar (9 pillars) |
| `section_entropy` | Shannon entropy of content sections — higher = broader reader |

> **Note:** The CTE alias `base_90` in the SQL is a legacy name from the 90-day version. It references the full 180-day window — the alias name is cosmetic only and does not affect results.

---

## 4. Feature Engineering

All of the following is computed in Python after the data pull.

### Time-of-Day Ratios
Raw page view counts are divided by the user's total page views to produce within-user ratios (sum to 1.0). This removes the effect of visit volume — a user who visits 100 times and 80% in the evening gets the same `ratio_evening` as someone who visits 5 times with 4 in the evening.

| Slot | Hours (UTC) | Derived feature |
|---|---|---|
| Morning | 05:00–10:59 | `ratio_morning` |
| Lunch | 11:00–14:59 | `ratio_lunch` |
| Evening | 15:00–20:59 | `ratio_evening` |
| Night | 21:00–04:59 | `ratio_night` |
| Weekend | Sat/Sun | `ratio_weekend` |

### M-Score Composite
Three engagement signals are combined into a single score for the M dimension:

```
m_composite = 0.50 × dwell_s_median_normalised
            + 0.30 × pv_per_visit_avg_normalised
            + 0.20 × deep_read_rate_normalised
```

Normalised using MinMaxScaler. Null dwell values (single-page visits) are filled with 0.

### Section Entropy
Computed in SQL as Shannon entropy over content sections. Clipped at 0 in Python to remove floating-point `-0.0` artefacts.

---

## 5. Stage 1 — RFM Scoring & House Assignment

### R Score (Recency) — calibrated for 180-day window

| R Score | Recency | Meaning |
|---|---|---|
| 1 | 0–7 days | Active this week |
| 2 | 8–30 days | Active this month |
| 3 | 31–90 days | Haven't visited in 1–3 months |
| 4 | 91–180 days | Haven't visited in 3–6 months |

### F Score (Frequency)
Quartile split of `total_visits_90d` (visit count over 180 days):

| F Score | Quartile |
|---|---|
| 1 | Top 25% most frequent visitors |
| 2 | 50th–75th percentile |
| 3 | 25th–50th percentile |
| 4 | Bottom 25% (lowest frequency) |

### M Score (Engagement)
Quartile split of `m_composite`. M=1 is highest engagement (long dwell, many pages, deep reads); M=4 is lowest.

### Special Flags (evaluated before RFM)

- **New:** `account_age_days <= 30` — joined in the last month
- **Reactivated:** `reactivation_gap_days IS NOT NULL` AND not new — returned after a 60+ day absence within the last 60 days

### House Assignment Rules (first match wins)

| House | Rule | Lifecycle meaning |
|---|---|---|
| **New** | `account_age_days ≤ 30` | Brand new user — no behavioural history yet |
| **Reactivated** | Returned after 60+ day gap | Lapsed user who came back — high intent signal |
| **Champions** | R=1, F∈{1,2}, M∈{1,2} | Best users: active, frequent, highly engaged |
| **Needs Attention** | R=2, F∈{1,2}, M∈{1,2} | Were Champions last month — starting to drift |
| **Loyalists** | R∈{1,2}, F∈{1,2}, M∈{3,4} | Visit often but skim — habit without depth |
| **Show Potential** | R∈{1,2}, F∈{3,4}, M∈{1,2} | Infrequent but deep when they visit |
| **Occasional Browsers** | R∈{1,2}, F∈{3,4}, M∈{3,4} | Recent but low frequency and low engagement |
| **At Risk - HV** | R=3, M∈{1,2} | High-value users not seen in 1–3 months |
| **At Risk - LV** | R=3, M∈{3,4} | Low-value users not seen in 1–3 months |
| **Sleeping Giants** | R=4, M∈{1,2} | High-engagement users gone quiet for 3–6 months |
| **Dormant** | R=4, M∈{3,4} | Low-engagement users not seen in 3–6 months |

---

## 6. Stage 2 — Within-House Clustering

The clustering runs **only** on content preference, device, and time-of-day features — not on any of the RFM signals that defined the house. Within a house those dimensions are already bounded, so K-means finds genuine content/behaviour structure.

### Clustering Features (16 total)

| Category | Features |
|---|---|
| Content pillar (9) | `frac_pillar_news`, `frac_pillar_audio`, `frac_pillar_culture`, `frac_pillar_business`, `frac_pillar_home`, `frac_pillar_video`, `frac_pillar_innovation`, `frac_pillar_travel`, `frac_pillar_future_planet` |
| Content breadth (1) | `section_entropy` |
| Device (1) | `frac_desktop` |
| Time-of-day (5) | `ratio_morning`, `ratio_lunch`, `ratio_evening`, `ratio_night`, `ratio_weekend` |

### Houses Targeted for Clustering

| House | Reason |
|---|---|
| Champions | High value; clear audio vs news split observed |
| Needs Attention | High value; recently drifting Champions |
| Loyalists | Habit readers — content splits are actionable |
| Show Potential | Infrequent but engaged — small house |
| At Risk - HV | Priority retention segment |
| Sleeping Giants | Large lapsed segment with strong engagement history |
| **Skipped: Dormant** | Homogeneous on content; defined purely by inactivity |
| **Skipped: At Risk - LV** | Low value; limited actionability |
| **Skipped: Occasional Browsers** | Too small and homogeneous |
| **Skipped: New / Reactivated** | Insufficient behavioural history for stable clusters |

### K Selection
For each house, K is scanned from 2 to min(6, √(n/50)). Best K = highest silhouette score. If best silhouette < 0.20, the house is kept as a single group (it's genuinely homogeneous on content/device/time).

---

## 7. What Each Persona Means

### Active Houses

---

#### Champions – News Heavy
> **"Your power readers. Desktop-first, news-obsessed, visit nearly every day."**

- Recency: 1–2 days | Visits: 50+ per 6 months | Dwell: ~65s median
- ~70% of page views are news pillar; ~80% on desktop
- This is the core audience — heavy, loyal news consumers who form the backbone of readership metrics
- **Action:** Personalise with breaking news alerts, newsletters, subscription upgrade prompts. These users are most likely to convert to paid if not already.

---

#### Champions – Morning / Audio
> **"Early-morning audio listeners. A niche but deeply loyal segment."**

- Recency: 1–2 days | Visits: 50+ per 6 months | Dwell: ~50s median
- ~90% audio pillar; morning is the dominant time slot; lower desktop fraction (~30%)
- Despite the label "Morning" (from the naming algorithm), this segment is primarily defined by audio consumption — they listen rather than read, likely during commutes or morning routines
- **Action:** Audio-first content recommendations, podcast promotion, push notifications timed to morning slots.

---

#### Needs Attention – News Heavy
> **"Lapsed Champions — high-value users who visited last month but haven't been back."**

- Recency: 8–30 days | Visits: 10–50 per 6 months | Dwell: ~70s median
- Very similar profile to Champions (news-dominant, desktop) but with a recency gap of 8–30 days
- These users were recently highly engaged and are at an inflection point — a small nudge can pull them back into Champions
- **Action:** Re-engagement email highlighting content they haven't seen, "what you missed" digest, personalised notifications.

---

#### Needs Attention – Audio
> **"Audio fans who are starting to drift."**

- Recency: 8–30 days | Visits: 10–15 per 6 months | Dwell: ~50s median
- ~84% audio pillar — the audio equivalent of the Needs Attention – News Heavy segment
- Smaller but very distinct — if audio content has been light recently, this may explain their drop-off
- **Action:** Promote new audio/podcast releases directly, audio-specific re-engagement.

---

#### Loyalists – News Heavy
> **"Habitual news skimmers — come back often but don't read deeply."**

- Recency: 0–7 days | Visits: 8–15 per 6 months | Dwell: ~25s median
- Frequent visitors with low dwell time — they check headlines but don't sit with articles
- High news fraction (~70%), moderate desktop usage
- **Action:** Homepage personalisation, quick-read formats, newsletter optimised for scan-reading. Don't push long-form — they won't read it.

---

#### Loyalists – Culture
> **"Regular visitors with a culture and arts focus — slightly deeper readers than the main Loyalist group."**

- Recency: 0–7 days | Visits: 5–10 per 6 months | Dwell: ~30s median
- Culture pillar is the distinguishing feature (~50% culture vs ~14% for the house average)
- Half their reading is still news, but culture content is their secondary interest
- **Action:** Culture section highlights, arts/entertainment newsletters, event-driven content.

---

#### Show Potential – [Time Variant]
> **"Rare but deep visitors — high engagement per session despite low frequency."**

- Recency: 8–21 days | Visits: 1–3 per 6 months | Dwell: 120–160s median | Deep read rate: ~17%
- These users read very deeply when they visit — dwell times are the highest of any persona
- The sub-clusters (Night Owl, Evening, Morning, Desktop, Home Page) reflect time-of-day preferences, not meaningful content differences — the house is too small for stable content splits
- **Action:** High-value long-form content, in-depth features, investigative pieces. These users have the appetite for premium content and are most likely to value a subscription.

---

### At-Risk Houses

---

#### At Risk - HV – News Heavy
> **"Your most valuable churning users — high-engagement news readers gone quiet for 1–3 months."**

- Recency: 31–90 days | Visits: 3–8 per 6 months | Dwell: ~70s median
- Excellent engagement metrics when active — comparable dwell and read depth to Champions
- The largest at-risk segment and the highest business priority: these are users worth paying to win back
- **Action:** Aggressive re-engagement campaign — personalised email with top stories from their favourite sections, win-back offer if behind a paywall, "we miss you" push notification.

---

#### At Risk - HV – General
> **"Mystery deep readers — high engagement but zero news consumption."**

- Recency: 31–90 days | Visits: 2–4 per 6 months | Dwell: ~175s median | Deep read rate: ~20%
- This is the most unusual sub-cluster: the longest dwell times and highest deep read rates of any persona, but with virtually no news consumption
- They are reading something very deeply — likely long-form features, investigative journalism, or niche pillars
- The generic "General" label means the clustering algorithm couldn't identify a single dominant pillar (their reading is spread across non-news content)
- **Action:** Investigate what sections these users actually visit (a follow-up query on content_section for this group would reveal it). Likely candidates: innovation, culture, future-planet. Target with long-form and investigative content.

---

#### At Risk - LV
> **"Low-engagement users who haven't visited in 1–3 months."**

- Recency: 31–90 days | Visits: 1–2 per 6 months | Dwell: ~22s median
- The second-largest segment in the 90-day run (~16%) and a consistently large group
- Low engagement when active — single-page visits, minimal dwell
- No sub-clustering applied (low value, high homogeneity)
- **Action:** Low-cost broad re-engagement (email blast, social). Don't invest heavily — conversion rate will be low. Focus on identifying if any of these users have a strong content affinity worth nurturing.

---

### Lapsed Houses

---

#### Sleeping Giants – Explorer
> **"High-engagement broad readers who have gone quiet for 3–6 months."**

- Recency: 91–180 days | Visits: 2–5 per 6 months | Dwell: ~80s median
- The "Explorer" label comes from high section entropy — these users read across many different topics and sections, not just one niche
- Strong engagement history makes them worth re-activating despite the long absence
- **Action:** "What's new" style re-engagement content showcasing breadth of coverage. Their broad reading habits mean a variety-focused approach will land better than niche-specific messages.

---

#### Sleeping Giants – Culture
> **"Lapsed culture enthusiasts — deep readers who have disappeared."**

- Recency: 91–180 days | Visits: 1–3 per 6 months | Dwell: ~150s median | Deep read rate: ~25%
- Almost no news consumption (~3.5%); culture is their primary interest
- Extremely deep readers when active — dwell of 150s is among the highest of any lapsed persona
- **Action:** Culture and arts re-engagement specifically. A new feature, interview, or editorial series in the culture section could pull these users back. Avoid news-focused re-engagement — it will miss them entirely.

---

#### Dormant
> **"Low-engagement users not seen in 3–6 months."**

- Recency: 91–180 days | Visits: 1 per 6 months | Dwell: ~22s median
- The largest or second-largest segment depending on window length
- Very low engagement signals — these users were never deeply invested in the product
- Not sub-clustered (homogeneous, low value)
- **Action:** Minimal investment. A single low-cost email to re-surface the brand. If no response, suppress from future campaigns to protect deliverability.

---

### Acquisition / Lifecycle Houses

---

#### New
> **"Users who joined in the last 30 days — still forming habits."**

- Account age: ≤ 30 days | Visits: 1–3 | Dwell: ~40s median
- Behavioural patterns are not yet stable — they're still exploring what the product offers
- Not sub-clustered (insufficient history for stable content clusters)
- **Action:** Onboarding flows, personalised content discovery, welcome emails highlighting the breadth of coverage. The goal is to convert a single-visit user into a habit. First-30-day behaviour strongly predicts long-term retention.

---

#### Reactivated
> **"Lapsed users who came back after a 60+ day absence in the last 60 days."**

- Recency: low (they returned recently) | Visits: 4–8 | Dwell: ~50s median
- High intent signal — something brought them back after a long absence (news event, campaign, referral)
- The 60-day gap threshold means these are genuine re-engagements, not just irregular-but-continuous visitors
- **Action:** Treat like New users with added personalisation based on their pre-lapse history. Strike while intent is high — a targeted re-onboarding sequence in the first 7 days back dramatically improves retention.

---

#### Occasional Browsers
> **"Recent but infrequent and low-engagement visitors."**

- Recency: 0–30 days | Visits: 1–2 | Dwell: ~20s median
- They arrived recently but didn't engage deeply — bounce-and-skim behaviour
- Likely driven by a specific link (social, search) rather than direct intent
- **Action:** Surface related content to extend the session. Homepage and article recommendation widgets are most relevant here.

---

## 8. Evaluation

### Silhouette Scores (from 90-day run — 180-day scores will differ slightly but pattern should hold)

| House | K | Silhouette | Quality |
|---|---|---|---|
| Needs Attention | 2 | 0.896 | Excellent |
| Champions | 2 | 0.879 | Excellent |
| At Risk - HV | 2 | 0.872 | Excellent |
| Sleeping Giants | 2 | 0.836 | Excellent |
| Loyalists | 2 | 0.648 | Good |
| Show Potential | 5 | 0.247 | Weak — borderline noise |

Scores above 0.80 indicate very clean cluster separation. The two-stage approach is the key reason — within a house, K-means only needs to find variation in content/device/time, not in the much larger engagement dimension.

### K=2 Wins in Most Houses
Almost every house selects K=2, indicating the dominant split is a single strong dimension: news vs audio in Champions and Needs Attention; news vs culture in Loyalists; news vs everything-else in At Risk and Sleeping Giants. This is not a failure of the model — it's a true reflection that news dominates the platform and the meaningful split is "news-first vs alternative content" within each lifecycle stage.

---

## 9. Known Issues & Limitations

### News Dominance
The largest cluster in every house is "News Heavy" (91–97% of users). News pillar fraction is so high across the user base that it compresses other content signals. The clustering is essentially separating news readers from everyone else.

**Fix in progress:** Applying `np.sqrt()` to `FRAC_PILLAR_NEWS` before clustering, or weighting news lower, would allow other pillars to surface more distinct clusters.

### Extreme Cluster Imbalance
All houses produce one large cluster (90–97%) and one small minority cluster. This is real — most users within a house share a similar content profile — but it limits actionability for the majority.

### Show Potential Clustering is Borderline
Silhouette of 0.247 with 5 clusters on ~2,000 users is not robust. The sub-clusters are splitting on time-of-day noise rather than real content structure. Consider removing Show Potential from `HOUSES_TO_CLUSTER`.

### CTE Named `base_90`
The SQL CTE is named `base_90` but queries 180 days of data. This is a legacy naming issue — it has no effect on results but should be renamed to `base_180` for clarity in a future cleanup.

### Column Named `total_visits_90d`
Similarly, the Snowflake column `total_visits_90d` actually counts visits over 180 days. The name is inherited from the 90-day version and should ideally be `total_visits_180d` — but changing it requires updating both the SQL and all Python references simultaneously.

### `account_age_days` is a Lower Bound
For users older than 180 days, `account_age_days` reflects only days since their first visit *in the window*, not their true account age. Any value > 30 correctly excludes them from the New house. True account age would require a join to a user registration table.

---

## 10. Next Steps

### Quick Wins
1. **Rename `base_90` → `base_180` and `total_visits_90d` → `total_visits_180d`** in the SQL and notebook to avoid confusion
2. **Remove Show Potential from `HOUSES_TO_CLUSTER`** — its silhouette is below a reliable threshold and the sub-clusters are not actionable
3. **Add At Risk - LV to clustering** — 213K users (15.9% of 90d run) sitting as an undifferentiated blob; there's likely a news vs non-news split worth finding

### Address News Dominance
4. **Dampen news signal before clustering:**
   ```python
   df['FRAC_PILLAR_NEWS_SQRT'] = np.sqrt(df['FRAC_PILLAR_NEWS'])
   ```
   Replace `FRAC_PILLAR_NEWS` with the sqrt version in `INTRA_FEATURES`. This compresses the range without removing the signal entirely.

5. **Fix the naming algorithm** — change from first-feature-above-threshold to highest-z-score-wins to prevent time-of-day features overriding content pillar names:
   ```python
   best_feat = max(
       (f for f in FEATURE_LABELS if f in z_scores.index and z_scores[f] > 0.35),
       key=lambda f: z_scores[f],
       default=None
   )
   ```

### Deeper Analysis
6. **Investigate At Risk - HV – General** — run a follow-up query on `content_section` for this sub-cluster to identify what they're actually reading. This mystery high-dwell, zero-news group likely represents a valuable niche audience.

7. **Add New users to clustering** — with 180d data, New users (joined in last 30 days) still have 30 days of behaviour. Split them into content-preference sub-groups to personalise onboarding earlier.

8. **Track persona migration over time** — run the pipeline weekly and compare house membership across runs. A user moving from Champions to Needs Attention is a churn early-warning signal; a Dormant user appearing in Reactivated is a win-back success.

### Production
9. **Write persona labels back to Snowflake** — output `df[['USER_ID', 'house', 'persona']]` to a gold-layer table so editorial, CRM, and product teams can query it directly without running the notebook.

10. **A/B test persona-driven interventions** — run separate re-engagement campaigns for At Risk - HV – News Heavy vs At Risk - HV – General and measure open rate, click rate, and return visit rate. The content-differentiated approach should outperform a single generic campaign.
