# ⚔️ Pokémon MCDA Lab (Streamlit)

Εφαρμογή Streamlit που:
1. **Pokédex & Stats** – φέρνει από το [PokeAPI](https://pokeapi.co) base stats, τύπους, abilities, κινήσεις και υπολογίζει τα stats σε οποιοδήποτε level (IV/EV/Nature).
2. **Damage Calculator** – τύπος ζημιάς Gen 9 (STAB, Tera, καιρός, terrain, crit, items, abilities, screens, spread), 16 rolls, πιθανότητα OHKO/2HKO…
3. **Κατάταξη Pokémon** – TOPSIS και/ή PROMETHEE II με επεξεργάσιμα κριτήρια, βάρη, κατεύθυνση και συναρτήσεις προτίμησης. Περιλαμβάνει πίνακα ζημιάς round-robin (κάθε Pokémon εναντίον όλων).
4. **Synergy & τετράδες** – δημιουργεί dataset συνέργειας για κάθε ζεύγος (αμυντική/επιθετική κάλυψη, ρόλοι, τύποι, abilities/καιρός), επιτρέπει χειροκίνητη διόρθωση ή upload δικού σας CSV, και κατατάσσει τις 15 τετράδες μιας 6άδας με TOPSIS/PROMETHEE.
5. **Μεθοδολογία** – όλοι οι τύποι.

Παιχνίδια: **Scarlet/Violet (+DLC)**, **Legends Z-A (+Mega Dimension)**, **Pokémon Champions**.

## Εγκατάσταση & εκτέλεση
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Η πρώτη φορά που ανοίγετε ένα Pokémon κατεβαίνουν τα δεδομένα του· μετά αποθηκεύονται στο `.cache/pokeapi/` και φορτώνουν αμέσως.
Χωρίς internet, διαλέξτε **Demo (offline)** στο sidebar (24 δημοφιλή Pokémon).

Tests: `python tests/test_core.py`

## Δομή
```
app.py              UI (Streamlit)
core/data.py        PokeAPI provider με disk cache + Demo provider
core/rosters.py     λίστες Pokémon ανά παιχνίδι
core/stats.py       τύποι stats, natures, έτοιμα spreads
core/damage.py      damage calculator
core/typechart.py   πίνακας τύπων
core/mcda.py        TOPSIS, PROMETHEE II
core/analysis.py    κριτήρια κατάταξης, συνέργεια, τετράδες
```

## Σημειώσεις / περιορισμοί
- Το PokeAPI δεν δημοσιεύει (ακόμα) pokedex για Z-A και Champions, οπότε οι λίστες τους είναι ενσωματωμένες στο `core/rosters.py` (Σεπτ. 2026) — προσθέστε νέα Pokémon εκεί όταν βγαίνουν seasons/DLC.
- Αν δεν υπάρχει learnset για Z-A/Champions στο PokeAPI, χρησιμοποιείται του Scarlet/Violet.
- Οι νέες Mega μορφές του Z-A/Champions εμφανίζονται μόνο όταν τις προσθέσει το PokeAPI.
- Το Z-A έχει real-time μάχες και καθόλου abilities· ο damage calculator χρησιμοποιεί τον κλασικό τύπο ως προσέγγιση.
