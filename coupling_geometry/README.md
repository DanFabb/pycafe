# Geometria di test per il coupling vibroacustico

Caso canonico: cavità acustica rigida a scatola + piastra flessibile su una faccia.

- `make_box_plate_mesh.py` — genera `box_plate.msh` (parametrico: `Lx, Ly, Lz, nx, ny, nz`).
- `explore_physical_groups.py` — carica la mesh con pyCAFE e verifica le proprietà su cui si basa la matrice di accoppiamento Kc.
- `run_two_domains.py` — i due domini separati e i loro modi disaccoppiati.
- `run_coupled.py` — **flusso completo**: Kc → modi accoppiati → risposta forzata → filmato 3D.

## Flusso di lavoro accoppiato

```python
system = pycafe.prepare_vibroacoustic_system(
    nodes=nodes, groups=groups, rho0=1.204, c0=343.0,
    t=0.002, rho_s=7800.0, E=210e9, nu=0.3,
)                                   # Kc costruita di default
freqs, modes = solve_vibroacoustic_modal(system, num_modes=6)
result = solve_vibroacoustic_frequency_sweep(
    system, frequencies, F_s=F_s, eta_s=0.02,
)
p_full = expand_pressure(result["p"], result["idx_a"], len(nodes))
```

### Convenzione di segno

Normale `n` **uscente dal fluido** (orientata automaticamente dall'elemento
fluido dietro ogni faccia), `Kc = ∫ Ns^T n Na dS`, quindi:

- forza sulla struttura `F_s = Kc p`;
- sorgente acustica `F_a = ρ₀ ω² Kc^T w` (velocità normale `v_n = jω n·w`).

Sistema non simmetrico della formulazione Euleriana `(w, p)`:

```
[ Ks + jωCs − ω²Ms        −Kc              ] {w}   {F_s}
[ −ρ₀ω² Kc^T        Ka + jωCa − ω²Ma       ] {p} = {F_a}
```

con `Mc = −ρ₀ Kc^T` della notazione classica. L'asimmetria non è perdita di
reciprocità: nasce dall'accoppiare uno spostamento con una pressione, per cui
un termine finisce nel blocco di rigidezza e l'altro in quello di massa.

### Verifica fisica (test_coupling.py)

| Controllo | Esito |
|---|---|
| `∫ n dS` da `Kc·1` | `[0, 0, 0.48] m²` = area piastra lungo +z |
| ρ₀ → 0 | riproduce piastra in vacuo + cavità rigida |
| aria, cavità sigillata | 1° modo 40.84 → **43.46 Hz** (+6.4%) |
| molla di cavità compatta `k = ρ₀c₀²(ΔV)²/V` | 43.70 Hz, scarto 0.5% |

Il modo a ~0 Hz del sistema accoppiato è la pressione uniforme della cavità
chiusa (equivalente acustico di un moto rigido): si elimina vincolando una
pressione con `pressure_zero_nodes0`.

## Costruzione

Scatola `0.8 × 0.6 × 0.5` m, mesh **transfinita** `8×6×5` → 240 HEXA8.
La piastra è la faccia superiore `z = Lz` del volume: **superficie condivisa**, non una superficie duplicata. Con `setTransfiniteSurface` + `setRecombine` su tutte le facce e `setTransfiniteVolume` sul volume, gmsh produce solo esaedri e quadrilateri strutturati.

Physical groups:

| Nome | Dim | Contenuto | Ruolo |
|---|---|---|---|
| `fluid` | 3 | 240 `Hexahedron 8` | dominio acustico (pressione) |
| `plate` | 2 | 48 `Quadrilateral 4` | dominio strutturale **e** interfaccia di coupling |
| `rigid_walls` | 2 | 188 `Quadrilateral 4` | pareti rigide (BC naturale) |
| `plate_clamp` | 1 | 28 `Line 2` | bordi piastra (incastro) |

Nota gmsh: con physical groups definiti, gmsh salva **solo** gli elementi appartenenti a un physical group (`Mesh.SaveAll=0` di default). Tutto ciò che serve deve stare in un gruppo.

## Findings (verificati da `explore_physical_groups.py`)

1. **Il loader pyCAFE riceve tre element types**: `Hexahedron 8` (240), `Quadrilateral 4` (236 = 48 piastra + 188 pareti, **in un unico array indistinto**), `Line 2` (28). Conseguenza: il dict `elements` da solo NON basta a distinguere piastra da pareti — serve la connettività per physical group (`load_mesh_with_groups` / `extract_physical_groups_with_connectivity`).
2. **Mesh conforme all'interfaccia**: i 63 nodi di `plate` sono un sottoinsieme dei 378 nodi di `fluid` — stessi tag gmsh, stessa numerazione. Niente mapping o interpolazione tra griglie: il coupling Kc si integra sulle facce condivise usando direttamente gli stessi indici nodali per pressione (fluido) e spostamento (struttura).
3. **Ogni QUAD4 della piastra è esattamente una faccia di un HEXA8** (48/48 match per set di nodi) → l'integrazione di superficie della Kc può usare le shape functions 2D del quad sulla faccia, coerenti con la traccia delle shape functions 3D dell'hexa.
4. **Normale della piastra**: `+z` con l'ordinamento nodi prodotto (verso l'esterno del fluido). Il segno della normale determina il segno della Kc (`−Nsᵀ·n·Na`): da fissare come convenzione quando si implementa il coupling.
5. **Tag nodali contigui 1..N**: l'assunzione del loader (`nodes = node_coords.reshape(-1,3)` con tag 1-based in ordine) regge su mesh transfinite. Da riverificare per mesh non strutturate/partizionate.
6. **Registry**: su questa mesh `find_acoustic_elements` seleziona `CHEXA8` (dim 3 vince sulle facce QUAD4). Il kernel non è ancora implementato → `NotImplementedError` finché non vengono fornite le funzioni di forma HEXA8.

## Convenzione physical groups proposta per i casi coupled

- `fluid` (dim 3): dominio acustico.
- `plate` / `structure` (dim 2): dominio strutturale; se coincide con l'interfaccia (caso tipico), fa da entrambi.
- `rigid_walls` (dim 2): pareti rigide; altri gruppi dim 2 liberi per impedenza/velocità (es. `inlet`, `absorbing`).
- `plate_clamp` (dim 1): vincoli strutturali sui bordi.

I gruppi senza nome vengono rinominati automaticamente `group_dim{d}_tag{t}` dal loader; i nomi duplicati ricevono suffisso `_dim{d}_tag{t}`.
