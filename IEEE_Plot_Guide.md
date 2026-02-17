# State-of-the-Art IEEE Plots — Vollständige Anleitung

## 1. Maße und Seitenlayout

- **Einzelspalte:** 3.5 inch (88.9 mm) breit
- **Doppelspalte:** 7.16 inch (181.6 mm) breit
- **Höhe:** So niedrig wie möglich, ohne Lesbarkeit zu opfern. Faustregel: Seitenverhältnis zwischen 3:1 (sehr kompakt, z.B. 3-Panel-Reihe) und 1.5:1 (Einzelplot). Niemals quadratisch oder hochkant.
- **Mehrteilige Figuren:** Immer `constrained_layout` statt `tight_layout` — es handhabt Titel, Legenden und Colorbars zuverlässiger.
- **DPI:** Mindestens 300 für Raster (PNG), 600 für Finaldruck. Vektorgrafiken (PDF/EPS) bevorzugen.

## 2. Schriften

- **Familie:** Serif (passt zu LaTeX-Dokumenten). In Matplotlib: `"font.family": "serif"`.
- **Basisgröße:** 8 pt. Das ist der Referenzwert — alles andere skaliert relativ dazu.
- **Achsen-Labels:** 9 pt (`"axes.labelsize": 9`)
- **Tick-Labels:** 7–8 pt (`"xtick.labelsize": 7.5`)
- **Legenden:** 7 pt (`"legend.fontsize": 7`)
- **Panel-Titel:** 8 pt, bold nur für die Panel-Nummer: `(a)` bold, Rest normal
- **Annotationen im Plot:** 6–7 pt (z.B. Contour-Labels, Inline-Beschriftungen)
- **Keine Font unter 5.5 pt** — wird im Druck unlesbar.
- **Konsistenz:** Alle Figuren im Paper müssen identische Font-Einstellungen verwenden. Ein zentrales `mpl.rcParams.update({...})` am Anfang jedes Skripts.

## 3. Farben

### Grundregeln
- **Niemals reines Schwarz (#000000)** für Linien — stattdessen dunkles Anthrazit (#2d3436). Wirkt weicher im Druck und auf Screen.
- **Niemals reines Rot/Blau/Grün** — immer leicht gedämpfte Varianten. Gute Defaults: Rot #d63031, Blau #0984e3, Grün #00b894, Orange #e17055, Violett #6c5ce7.
- **Maximal 4–5 Farben pro Panel.** Mehr ist visuell nicht unterscheidbar.
- **Farbenblind-tauglich:** Niemals Rot vs. Grün als einziges Unterscheidungsmerkmal. Immer zusätzlich Marker-Shape oder Linestyle variieren. Tools: [ColorBrewer](https://colorbrewer2.org), Matplotlib `tab10` ist akzeptabel.

### Divergierende Colormaps (für Phasendiagramme, Heatmaps)
- `RdBu_r` oder `coolwarm` mit `TwoSlopeNorm(vcenter=0)` — Weiß liegt exakt bei Null.
- Symmetrischer Bereich: `vmin = -vmax`.
- Mindestens 15–21 Contour-Levels für glatte Farbübergänge.

### Sequenzielle Colormaps
- `viridis`, `magma`, `inferno` — niemals `jet` (nicht farbenblind-tauglich, nicht wahrnehmungslinear).

## 4. Linien und Marker

- **Linienbreite:** 1.0–1.4 (Default 1.5 ist zu dick für IEEE-Spaltenbreite).
- **Marker-Größe:** 3–5 pt (`"lines.markersize": 4`). Default 6 ist zu groß.
- **Offene Marker bevorzugen:** `markerfacecolor="white"` mit `markeredgewidth=0.8`. Spart visuelles Gewicht, macht Überlappungen lesbar.
- **Jede Linie braucht ein eigenes Erkennungsmerkmal** — nie nur Farbe. Mindestens zwei von drei variieren: Farbe, Marker-Shape, Linestyle.
- **Marker-Shapes zuweisen nach Rolle:**
  - Kreise (o) für Baselines/Standard
  - Quadrate (s) für Kontrollbedingungen
  - Rauten (D) für die vorgeschlagene Methode
- **Gestrichelte Linien** (`--`) nur für Kontrollbedingungen oder theoretische Fits, nie für die Hauptmethode.

## 5. Confidence Bands und Fehlerbalken

- **Confidence Bands:** `alpha` zwischen 0.10 und 0.15. Niemals über 0.25 — das erschlägt die Linien.
- **Keine Randlinie** am Band: `linewidth=0` im `fill_between`.
- **Fehlerbalken statt Bands** wenn weniger als 8 Datenpunkte — Bands sehen bei wenigen Punkten künstlich interpoliert aus.
- **Cap-Size** für Fehlerbalken: 2–3 pt, `capthick=0.8`.
- **Immer angeben** was dargestellt wird: ±1 std, 95% CI, min/max. In der Caption, nicht im Plot.

## 6. Achsen

### Spines (Rahmenlinien)
- **Obere und rechte Spine entfernen:** `ax.spines["top"].set_visible(False)` und `ax.spines["right"].set_visible(False)`. Das ist Standard in Nature/Science/Top-Tier-Plots.
- **Achsenlinienbreite:** 0.5–0.6 pt.

### Ticks
- **Richtung nach innen:** `ax.tick_params(direction="in")`. Ticks nach außen sind veraltet.
- **Tick-Breite:** 0.4–0.5 pt (`"xtick.major.width": 0.5`).
- **Minor Ticks** nur wenn nötig (z.B. Log-Achsen). Sonst weglassen.
- **Keine Tick-Überlappung zwischen benachbarten Panels.** Letzte Tick-Label des linken Panels und erste des rechten entfernen wenn nötig.

### Grid
- **Dezent oder gar nicht.** Wenn Grid, dann: `linewidth=0.3`, `alpha=0.3–0.4`.
- **Nie beides:** Wenn volles Grid, dann keine Minor Ticks. Wenn Minor Ticks, dann kein Grid.

### Shared Axes
- **Shared y-Achse** bei nebeneinander liegenden Panels mit gleicher Metrik. Label nur am linken Panel.
- **Shared x-Achse** bei untereinander liegenden Panels. Label nur am untersten.

## 7. Legenden

- **Kein Rahmen:** `frameon=False`. Immer.
- **Innerhalb des Plots** positionieren, in einer leeren Ecke. Nie außerhalb (verschwendet Platz).
- **Kurze Labels:** "EV", "Ours", "$Q=6$" — nicht "Expectation-value training protocol".
- **Reihenfolge** der Einträge = Reihenfolge der visuellen Prominenz (die Hauptmethode zuerst oder die Baseline zuerst — konsistent bleiben).
- **Maximal 4 Einträge.** Mehr → andere Darstellungsform wählen (z.B. Farblegende neben dem Plot).
- **Keine Legende wenn überflüssig** — z.B. bei nur einer Linie oder wenn die Caption es erklärt.

## 8. Panel-Beschriftung

- **Labels als Titel über dem Panel:** `ax.set_title("(a) ...", loc="left", fontsize=8, fontweight="bold")`.
- **Nicht als Text-Annotation im Plot** — das kollidiert mit Daten.
- **Format:** `(a)` bold, beschreibender Text normal. Beispiel: **(a)** λ robustness
- **Konsistente Benennung** im ganzen Paper: immer (a), (b), (c) oder immer (A), (B), (C). Nie mischen.

## 9. Colorbars

- **Schmal:** Breite 0.012–0.02 der Figurbreite.
- **Neben dem letzten Panel**, nicht darunter (spart vertikalen Platz).
- **Outline-Breite:** 0.4 pt.
- **Tick-Label:** 6–7 pt.
- **Label am Colorbar:** 7–8 pt, möglichst kurz. Vollständige Beschreibung in die Caption.

## 10. Captions (Bildunterschriften)

- **Kein Bold im Caption-Text** außer dem Figurlabel ("Fig. 1.").
- **Struktur:** Ein Satz was dargestellt wird → Ein Satz was man sieht → Ein Satz was es bedeutet.
- **Panel-Referenzen** wenn nötig: "Left: ..., Right: ..." oder "(a) shows ..., (b) shows ...".
- **Alle Abkürzungen und Symbole** im Caption erklären, auch wenn sie im Text definiert sind — Figuren werden oft isoliert gelesen.
- **Keine Interpretation** die über die Daten hinausgeht — das gehört in den Fließtext.

## 11. Spezialregeln für bestimmte Plot-Typen

### Zeitreihen-Traces
- Ground Truth immer in Schwarz/Anthrazit, durchgezogen, leicht dicker (1.5–1.8).
- Vorhersagen in Farbe, dünner (1.0–1.3).
- Wenn mehrere Methoden: gestrichelt für Baselines, durchgezogen für eigene Methode.

### Heatmaps / Phasendiagramme
- `contourf` mit mindestens 15 Levels für glatte Farbübergänge.
- Datenpunkte als kleine Scatter-Marker (s=6–10, weiße Edge, alpha=0.7) auf der Heatmap.
- Contour-Linien für theoretische Grenzen: gestrichelt, dünn (0.6–0.8), Farbe die sich abhebt.

### Bar Charts
- **Nie 3D.** Nie Schatten. Nie Gradient-Fills.
- Edge-Farbe leicht dunkler als Fill, linewidth 0.5.
- Fehlerbalken in Schwarz, cap_size 2.

### Scatter Plots
- `alpha=0.5–0.7` wenn viele Punkte überlappen.
- Edge immer sichtbar: `edgecolors="white"`, `linewidths=0.3`.

## 12. Export-Einstellungen

- **PDF** für Vektorgrafiken (Standard für LaTeX/IEEE). `fig.savefig("fig.pdf")`.
- **PNG** nur wenn Raster nötig (z.B. Heatmaps mit vielen Levels). Dann DPI ≥ 300.
- **`bbox="tight"`** immer setzen: `"savefig.bbox": "tight"`.
- **Keine Transparency** in Finaldruck-PDFs — manche IEEE-Workflows flatten das falsch. Alpha im Plot ist okay (wird gerastert), aber keine transparenten Hintergründe.
- **Weißer Hintergrund** explizit setzen: `fig.patch.set_facecolor("white")`.

## 13. Matplotlib rcParams Template

```python
import matplotlib as mpl

mpl.rcParams.update({
    # Schriften
    "font.family":        "serif",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    7,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,

    # Linien und Marker
    "lines.linewidth":    1.3,
    "lines.markersize":   4,

    # Achsen
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",

    # Grid (optional)
    "axes.grid":          False,
    "grid.linewidth":     0.3,
    "grid.alpha":         0.4,

    # Export
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "figure.constrained_layout.use": True,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",

    # Legende
    "legend.frameon":     False,
})
```

## 14. Checkliste vor Submission

Jeden Plot vor Einreichung gegen diese Liste prüfen:

1. Sind alle Texte bei gedruckter Spaltengröße lesbar (≥ 5.5 pt)?
2. Ist die Figur bei Schwarz-Weiß-Druck noch unterscheidbar (Marker + Linestyle)?
3. Sind obere und rechte Spines entfernt?
4. Sind Legenden rahmenlos und innerhalb des Plots?
5. Sind Confidence Bands dezent (alpha ≤ 0.15)?
6. Sind Panel-Labels konsistent ((a), (b), (c)) und als Titel formatiert?
7. Stimmen die Einheiten und Symbole in den Achsen-Labels?
8. Ist die Caption selbsterklärend ohne den Fließtext?
9. Sind alle Farben farbenblind-tauglich?
10. Exportiert als Vektor-PDF (oder PNG ≥ 300 DPI)?
