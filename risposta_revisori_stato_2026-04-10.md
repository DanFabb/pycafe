# Risposta ai Revisori — Stato Avanzamento
**Data:** 10 aprile 2026  
**Progetto:** pyCAFE — A Finite Element Framework for Solving Acoustic Problems in Python  
**Rivista:** Journal of Open Research Software (JORS)  
**Revisori:** 7 (R1–R7)

---

## ✅ FATTO

### Validazione analitica e benchmark (R3, R4, R5)

- **`examples/analytical_validation.ipynb`**  
  Confronto completo delle frequenze proprie CQUAD8 vs soluzione analitica esatta per cavità rettangolare rigida 2D (1 m × 0.5 m, aria, pareti rigide). Tabella errori modo per modo. Benchmark classico della letteratura (Helmholtz 2D con Neumann omogeneo).

- **`examples/convergence_hp.ipynb`** e **`examples/pycafe_convergence_hp.ipynb`**  
  Studio di h-convergenza (Nx = 2 … 64) e p-convergenza (CQUAD4 vs CQUAD8) con tassi di convergenza misurati tramite regressione log-log nella zona asintotica. Risultati: CQUAD4 → slope ≈ 2, CQUAD8 → slope ≈ 4, in accordo con la teoria FEM per problemi agli autovalori con soluzione regolare.

- **`examples/comparison_pycafe_fenics.ipynb`**  
  Confronto diretto pyCAFE (CQUAD8) vs FEniCSx (Q2, stessa densità di griglia): entrambi convergono all'analitico con errore < 0.001%. Risponde al punto di novelty positioning sollevato da R3 e R5.

- **`examples/comparison_20modes.ipynb`**  
  Confronto su 20 modi acustici con criterio di raffinamento mesh (6 CQUAD8 per lunghezza d'onda al modo più alto).

### Suite di test espansa (R1, R3)

- **`tests/test_element_matrices.py`**  
  Test unitari su matrici elementari K_e e M_e: simmetria, K·1 = 0 (modo a pressione uniforme), conservazione della massa, partizione dell'unità delle funzioni di forma, delta di Kronecker ai nodi.

- **`tests/test_solver_modal.py`**  
  Test di validazione del solver modale vs soluzione analitica con tolleranze mesh-dipendenti, convergenza monotona al raffinamento, ortogonalità delle mode shapes rispetto alla matrice di massa.

- **`tests/test_boundary_conditions.py`**  
  Test unitari su eliminazione di Dirichlet, mappatura idx_free, conservazione della simmetria delle matrici ridotte.

- **`tests/test_solver_helmholtz.py`**  
  Test del solver Helmholtz nel dominio della frequenza.

### CI Pipeline — GitHub Actions (R3)

- **`.github/workflows/ci.yml`**  
  Workflow di Continuous Integration che si attiva automaticamente ad ogni `git push` su `main` e ad ogni Pull Request. Esegue `pytest tests/ -v` su una macchina virtuale pulita con Python 3.10, 3.11 e 3.12 in parallelo, verificando la compatibilità del codice su tutte le versioni supportate. Installa le dipendenze da zero (numpy, scipy, matplotlib, tqdm, gmsh) per garantire riproducibilità su ambienti esterni.

### Riproducibilità e getting started (R2, R1)

- **`examples/analisi_modale_guidata.ipynb`**  
  Notebook step-by-step per nuovi utenti, con spiegazioni in prosa tra ogni cella di codice.

### Zenodo DOI (R6, R7) — 15 aprile 2026

- **DOI Zenodo depositato** il 15 aprile 2026: identificatore persistente del software ottenuto e da inserire nel paper. Risponde al requisito compulsory sollevato da R6 e R7.

---

## ❌ MANCANTE — DA FARE

### Compulsory (bloccanti per l'accettazione)

| # | Item | Reviewer | Note |
|---|---|---|---|
| 1 | ~~**Zenodo DOI**~~ | R6, R7 | ✅ Depositato il 15 aprile 2026 — da fare: aggiornare il paper con il DOI |
| 2 | **Identificare il software FEM commerciale** usato nella validazione originale | R6 | Aggiungere nome e versione del software commerciale citato |
| 3 | **Typo "Heigth" → "Height"** in Figure 2 del paper | R6 | Fix banale nel sorgente del paper |
| 4 | **Discutere alternative open-source** e pacchetti Python per l'acustica | R7 | Aggiungere paragrafo in introduzione: FEniCSx, scikit-fem, openCFS, ecc. |
| 5 | **Aggiungere 5–7 keyword** ricercabili nel dominio | R1 | Es. "Helmholtz equation", "frequency domain", "modal analysis", "CQUAD8", "computational acoustics" |

### Major

| # | Item | Reviewer | Note |
|---|---|---|---|
| 6 | ~~**CI pipeline GitHub Actions**~~ | R3 | ✅ Creato `.github/workflows/ci.yml` — da fare: `git push` per attivare |
| 7 | **Roadmap nel paper** — 3D, time-domain, PML, altri elementi | R3, R4, R5 | Aggiungere sezione "Future Work" |
| 8 | **Abstract** — rimuovere tono promozionale, essere più tecnici | R1 | Riscrivere indicando esplicitamente: elementi, equazione, domini di applicazione |
| 9 | **Introduzione** — confronto con strumenti esistenti + citazioni | R1, R7 | Tabella o paragrafo che posiziona pyCAFE vs FEniCSx, openCFS, MATLAB toolbox |
| 10 | **Sezione Quality Control nel paper** — espandere | R1, R3 | I test esistono ma il paper non descrive cosa testano e i risultati |
| 11 | **Discussione performance nel paper** — DOF vs runtime con numeri | R3 | I dati ci sono nei notebook di convergenza, vanno sintetizzati nel paper |
| 12 | **Novelty vs FEniCS/MATLAB-CAFE** — posizionamento chiaro nel testo | R3, R5 | Spiegare cosa pyCAFE aggiunge/semplifica rispetto ai framework generali |

### Minor

| # | Item | Reviewer | Note |
|---|---|---|---|
| 13 | Fix Reference [6] — nomi autori mancanti nella bibliografia | R6 | |
| 14 | PyPI submission | R6 | Pubblicare il pacchetto su PyPI |
| 15 | Figure 3 — aumentare font size | R7 | |
| 16 | Chiarire "low to mid frequency range" nel paper con limiti espliciti | R7 | |
| 17 | Esplicitare nel paper che pyCAFE è **2D only** (limitazione attuale) | R7 | |
| 18 | Aggiungere comando `pip install pycafe` nel paper | R6 | |
| 19 | Emoji in `test_basic.py` — intenzionale? Chiarire o rimuovere | R7 | |

---

## Priorità suggerite

1. **Prima:** items 1–5 (compulsory) — senza questi l'editor non procede
2. **Poi:** fare `git push` per attivare la CI già creata + items 10–11 (aggiornare testo paper con test e performance)
3. **Quindi:** items 7–9, 12 (revisione sostanziale del testo: abstract, intro, roadmap, novelty)
4. **Infine:** items 13–19 (minor, veloci)

---

*Documento generato il 10 aprile 2026*
