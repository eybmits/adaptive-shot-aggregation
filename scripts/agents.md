# agens.md — Verbesserungen für Fig. 2 (Benchmarks) & Storyline „Traj vs. Grouped‑Traj“

**Ziel:** Die Abbildungen und Auswertungen so überarbeiten, dass wir *wissenschaftlich sauber* zeigen können  
1) **Traj bringt etwas** (gegenüber EV),  
2) **Grouped‑Traj bringt etwas** (gegenüber EV und/oder Traj),  
3) **es einen Trade‑off** zwischen *NRMSE (Genauigkeit)* und *Amplitude/Varianz (Std‑ratio)* gibt — und dass wir eine **Konfiguration finden**, die **beides** gut macht (gute Amplitude *und* guter NRMSE).

---

## 0) Ausgangslage (was aktuell auffällt)

In Fig. 2 sieht man:

- **Mackey‑Glass:** NRMSE‑Vorteil von G‑Traj ggü. EV ist klein (**Δ≈+0.021**), aber sichtbar. Std‑ratio ist bei EV ~1, bei Traj sehr klein (~0.22), bei G‑Traj moderat (~0.76).  
- **NARMA‑10:** NRMSE‑Vorteil von G‑Traj ggü. EV ist größer (**Δ≈+0.140**), **aber** Std‑ratio bricht für Traj und G‑Traj stark ein (≈0.09 bzw. ≈0.13) → **Amplitude/Varianz kollabiert** (Signal wird zu „zu glatt/zu konstant“).  

Das heißt: Wir zeigen aktuell zwar „NRMSE wird besser“, aber bei NARMA wirkt das wie „mean predictor gewinnt“ und nicht wie „gute Dynamik wird rekonstruiert“.

---

## 1) Wissenschaftliche Leitplanken (damit es nicht nach Cherry‑Picking aussieht)

**Wichtig:** Den „besten“ Plot‑Ausschnitt (Test time steps) dürfen wir nicht per Auge wählen.

**Regel:** Der gezeigte Zeit‑Ausschnitt muss über ein *vordefiniertes, datengetriebenes* Kriterium gewählt werden, das **nicht** auf „Traj vs. G‑Traj sieht besser aus“ optimiert.

Empfohlene, saubere Kriterien (eine auswählen und dokumentieren):

1. **EV‑Worst‑Window (empfohlen):** Wähle das Fenster, in dem **EV** die größte (N)RMSE hat.  
   → Auswahl nutzt nur (True, EV). Traj/G‑Traj bleiben „unseen“ bei der Auswahl.
2. **High‑Variance‑Window:** Wähle das Fenster mit maximaler Varianz von **True**.  
   → zeigt Amplitude/Std‑ratio besonders gut.
3. **Fixed Windows:** z.B. drei feste Fenster bei 10%, 50%, 90% der Testlänge.  
   → extrem robust gegen Bias, aber evtl. weniger „visuell aussagekräftig“.

---

## 2) Konkreter Vorschlag: „repräsentativer Test time step“

### 2.1 Empfohlene Fensterlängen
- **Mackey‑Glass:** `L = 50–80` Schritte (Signal ist „smooth“, längeres Fenster macht Sinn)
- **NARMA‑10:** `L = 80–120` Schritte (hochfrequent/noisy → Fenster etwas länger, aber nicht das ganze T=300)

### 2.2 Praktische Defaults (wenn ihr *sofort* einen Plot braucht)
> Diese Werte sind visuell aus der aktuellen Fig. 2 plausibel — sollten aber **durch 2.3** ersetzt werden.

- **Mackey‑Glass:** **t = 85…125**  
  (dort hat EV den auffälligen „Dip“ um ~100, G‑Traj bleibt näher an True)
- **NARMA‑10:** **t = 60…120**  
  (dort sieht man mehrere True‑Spikes und dass G‑Traj/Traj zu flach werden)

### 2.3 Algorithmische Auswahl (empfohlen)
Implementiere eine Funktion, die ein Fenster automatisch auswählt:

**Input:** `y_true[t]`, `y_ev[t]`, Fensterlänge `L`  
**Score pro Fensterstart s:**  
- `err_ev(s) = NRMSE(y_true[s:s+L], y_ev[s:s+L])`  
- optional Stabilisierung: `var_true(s) = std(y_true[s:s+L])`  
- **Score:** `score(s) = err_ev(s)` oder `err_ev(s) * var_true(s)`  

**Output:** `s* = argmax_s score(s)` und `window = [s*, s*+L)`

> Wichtig: Das Fenster wird **nur** über EV ausgewählt. Danach plotten wir alle Methoden im selben Fenster.

---

## 3) Fig. 2: Layout‑Fix (Plot d ist zu kompakt)

### 3.1 Grundproblem
- NARMA‑10 ist hochfrequent. In der aktuellen Größe mit T=300 werden Linien zu „schwarzer Wand“.
- Außerdem fehlen Traj‑Kurven in den Timeseries‑Plots (aktuell nur EV & G‑Traj sichtbar) → Storyline „Traj vs G‑Traj“ wird nicht getragen.

### 3.2 Neue Plot‑Struktur (empfohlen)

**Option A (beste Lesbarkeit):** Split in zwei Figuren
- **Fig. 2:** Mackey‑Glass (Timeseries + NRMSE + Std‑ratio)
- **Fig. 3:** NARMA‑10 (Timeseries + NRMSE + Std‑ratio)

**Option B (eine Figure behalten, aber entschlacken):**
- 2 Zeilen (MG / NARMA)
- 3 Spalten, aber **Timeseries bekommt ~2× Breite** (GridSpec: `[2,1,1]`)
- Zusätzlich: **Zoom‑Inset** oder **nur ausgewähltes Fenster** plotten.

### 3.3 Timeseries‑Design (damit Traj & G‑Traj sichtbar werden)
- Plot nicht alle Methoden in eine Achse „quetschen“, sondern **Small‑Multiples**:
  - z.B. drei Mini‑Panels pro Dataset:  
    **True vs EV**, **True vs Traj**, **True vs G‑Traj** (gleiche y‑Achse!)
- Oder: Overlay behalten, aber:
  - True: schwarz, dick  
  - EV/Traj/G‑Traj: dünner, mit `alpha≈0.8`, klarer Legend
  - Optional: markiere das repräsentative Fenster (Hintergrund‑Shading)

### 3.4 Für NARMA‑10 speziell
- Plot **nur das repräsentative Fenster** (z.B. 80–120 steps), nicht T=300.
- Alternativ: Downsample (jeden 2.–3. Schritt) **nur für die Visualisierung**.
- Ergänzend (sehr hilfreich):  
  **Scatter `y_true` vs `y_pred`** (zeigt amplitude collapse sofort) oder Histogramm‑Overlay.

---

## 4) Std‑ratio wieder „gut“ bekommen (besonders für NARMA & G‑Traj)

### 4.1 Diagnose‑Checkliste (erst Bugs ausschließen)
1. **Normalisierung/Denormalisierung konsistent?**  
   - Wird `y_pred` in *denselben* Einheiten wie `y_true` zurückskaliert?
2. **Std‑ratio Definition korrekt?**  
   - `std_ratio = std(y_pred_test) / std(y_true_test)`  
   - über denselben Zeitbereich wie NRMSE.
3. **Clipping / Saturation?**  
   - Werden Outputs begrenzt (z.B. activation saturation), sodass Varianz sinkt?
4. **Grouped‑Traj‑Averaging?**  
   - Falls ihr Trajektorien mittelt, ist Varianzverlust „by design“.  
     → dann braucht ihr *eine Korrektur* oder *andere Gruppierung*.

### 4.2 Minimal‑Fix: Affine Kalibrierung (fair & sehr effektiv)
Wenn das Modell die Form trifft, aber zu kleine Amplitude hat:

- Fit auf **Validation** (nicht Test!) eine affine Korrektur:
  - `y_cal = a * y_pred + b`
  - `a = std(y_val_true) / std(y_val_pred)`
  - `b = mean(y_val_true) - a * mean(y_val_pred)`
- Reporte Metriken **vor/nach** Kalibrierung.

Vorteil: Schnell, interpretiert, behebt Std‑ratio systematisch.

### 4.3 Trainings‑Fix: Amplituden‑Regularisierung (zeigt Trade‑off sauber)
Erweitere Loss:

- `L_total = L_NRMSE + λ * (std_ratio - 1)^2`  
  oder numerisch stabil: `λ * (log(std_ratio))^2`

Sweep über `λ` → erzeugt **Pareto‑Front** (NRMSE vs Std‑ratio).  
Das ist *perfekt* für die Paper‑Story „Trade‑off“.

### 4.4 Gruppierung/Traj‑Design‑Fixes (wenn G‑Traj Varianz „wegmittelt“)
- **Kleinere Gruppen** (weniger Aggressives Clustering)
- **Mixture statt Mittelwert:** pro Gruppe ein Expert, Output ist gewichtete Mischung
- **Group‑wise scaling:** jede Gruppe bekommt eigene (a,b)‑Kalibrierung (auf Val)
- **Cluster‑Feature prüfen:** wenn ihr im „level“ clustert, verliert ihr Varianz.  
  Besser: in z‑scored/shape‑space clustern (z.B. normierte Segmente).

---

## 5) Wie wir die Story wissenschaftlich „wasserdicht“ erzählen

### 5.1 Reporte immer zwei Achsen
- **NRMSE** (Accuracy)
- **Std‑ratio** (Amplitude / Dynamik)

Optional (bei NARMA hilfreich):
- Pearson‑Korrelation
- R²
- MSE der *z‑scored* Serien (shape‑fit)

### 5.2 Zeige Trade‑off explizit
- Ein Plot: **NRMSE vs Std‑ratio** für EV, Traj, G‑Traj (und ggf. verschiedene λ / group size)  
  → Das zeigt *sofort* ob G‑Traj „overall better“ oder nur „glatter“.

### 5.3 Baseline gegen „Mean Predictor“
Für NARMA ist das wichtig:
- Baseline: `y_pred(t) = mean(y_train)`  
Wenn Traj/G‑Traj nur minimal besser als Mean‑Baseline ist, muss das offen diskutiert werden.

---

## 6) Definition von „Erfolg“ (Acceptance Criteria)

Wir nennen eine Konfiguration „publizierbar“, wenn:

1. **NRMSE verbessert** ggü. EV (im Mittel über Seeds)  
2. **Std‑ratio nicht kollabiert**: z.B. `0.8 ≤ std_ratio ≤ 1.2`  
3. Die **Timeseries‑Plots** zeigen klar:  
   - wo EV abweicht  
   - dass Traj oder G‑Traj *systematisch* näher an True ist (nicht nur zufällig)

---

## 7) Deliverables (was der Agent liefern soll)

1. Script/Funktion: `select_representative_window(...)`
2. Neue Figure(s):
   - MG: full + zoom / small‑multiples
   - NARMA: zoomed window + optional scatter/hist
3. Pareto‑Plot: (NRMSE, Std‑ratio) für EV/Traj/G‑Traj + Sweeps (λ oder group size)
4. Kurzer Textbaustein (Paper): „Trade‑off + warum unser Fenster objektiv gewählt ist“

