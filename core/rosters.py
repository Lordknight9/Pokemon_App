"""Game rosters.

* Scarlet/Violet: pulled live from PokeAPI pokedexes (paldea + kitakami + blueberry).
* Legends Z-A and Pokémon Champions: PokeAPI does not (yet) publish a pokedex for
  them, so the lists are bundled here (sources: pokemondb.net Lumiose + Mega Dimension
  dex, onemorecatch.site Champions roster, Sept 2026). Update the lists when new
  seasons/DLC add Pokémon.
"""
import re
import unicodedata

ZA_LUMIOSE = """Chikorita, Bayleef, Meganium, Tepig, Pignite, Emboar, Totodile, Croconaw, Feraligatr, Fletchling,
Fletchinder, Talonflame, Bunnelby, Diggersby, Scatterbug, Spewpa, Vivillon, Weedle, Kakuna, Beedrill, Pidgey,
Pidgeotto, Pidgeot, Mareep, Flaaffy, Ampharos, Patrat, Watchog, Budew, Roselia, Roserade, Magikarp, Gyarados,
Binacle, Barbaracle, Staryu, Starmie, Flabébé, Floette, Florges, Skiddo, Gogoat, Espurr, Meowstic, Litleo, Pyroar,
Pancham, Pangoro, Trubbish, Garbodor, Dedenne, Pichu, Pikachu, Raichu, Alolan Raichu, Cleffa, Clefairy, Clefable,
Spinarak, Ariados, Ekans, Arbok, Abra, Kadabra, Alakazam, Gastly, Haunter, Gengar, Venipede, Whirlipede, Scolipede,
Honedge, Doublade, Aegislash, Bellsprout, Weepinbell, Victreebel, Pansage, Simisage, Pansear, Simisear, Panpour,
Simipour, Meditite, Medicham, Electrike, Manectric, Ralts, Kirlia, Gardevoir, Gallade, Houndour, Houndoom, Swablu,
Altaria, Audino, Spritzee, Aromatisse, Swirlix, Slurpuff, Eevee, Vaporeon, Jolteon, Flareon, Espeon, Umbreon,
Leafeon, Glaceon, Sylveon, Buneary, Lopunny, Shuppet, Banette, Vanillite, Vanillish, Vanilluxe, Numel, Camerupt,
Hippopotas, Hippowdon, Drilbur, Excadrill, Sandile, Krokorok, Krookodile, Machop, Machoke, Machamp, Gible, Gabite,
Garchomp, Carbink, Sableye, Mawile, Absol, Riolu, Lucario, Slowpoke, Galarian Slowpoke, Slowbro, Galarian Slowbro,
Slowking, Galarian Slowking, Carvanha, Sharpedo, Tynamo, Eelektrik, Eelektross, Dratini, Dragonair, Dragonite,
Bulbasaur, Ivysaur, Venusaur, Charmander, Charmeleon, Charizard, Squirtle, Wartortle, Blastoise, Stunfisk,
Galarian Stunfisk, Furfrou, Inkay, Malamar, Skrelp, Dragalge, Clauncher, Clawitzer, Goomy, Sliggoo, Goodra,
Delibird, Snorunt, Glalie, Froslass, Snover, Abomasnow, Bergmite, Avalugg, Scyther, Scizor, Pinsir, Heracross,
Emolga, Hawlucha, Phantump, Trevenant, Scraggy, Scrafty, Noibat, Noivern, Klefki, Litwick, Lampent, Chandelure,
Aerodactyl, Tyrunt, Tyrantrum, Amaura, Aurorus, Onix, Steelix, Aron, Lairon, Aggron, Helioptile, Heliolisk,
Pumpkaboo, Gourgeist, Larvitar, Pupitar, Tyranitar, Froakie, Frogadier, Greninja, Falinks, Chespin, Quilladin,
Chesnaught, Skarmory, Fennekin, Braixen, Delphox, Bagon, Shelgon, Salamence, Kangaskhan, Drampa, Beldum, Metang,
Metagross, Xerneas, Yveltal, Zygarde, Diancie, Mewtwo"""

ZA_HYPERSPACE = """Mankey, Primeape, Annihilape, Meowth, Alolan Meowth, Galarian Meowth, Persian, Alolan Persian,
Perrserker, Farfetch'd, Galarian Farfetch'd, Sirfetch'd, Cubone, Marowak, Alolan Marowak, Porygon, Porygon2,
Porygon-Z, Capsakid, Scovillain, Tinkatink, Tinkatuff, Tinkaton, Cyclizar, Glimmet, Glimmora, Rotom, Greavard,
Houndstone, Sandygast, Palossand, Kecleon, Flamigo, Cryogonal, Dondozo, Tatsugiri, Frigibax, Arctibax, Baxcalibur,
Gimmighoul, Gholdengo, Qwilfish, Hisuian Qwilfish, Overqwil, Treecko, Grovyle, Sceptile, Torchic, Combusken,
Blaziken, Mudkip, Marshtomp, Swampert, Feebas, Milotic, Chingling, Chimecho, Indeedee, Purrloin, Liepard, Munna,
Musharna, Throh, Sawk, Yamask, Cofagrigus, Runerigus, Wimpod, Golisopod, Nickit, Thievul, Clobbopus, Grapploct,
Mimikyu, Kleavor, Morpeko, Golett, Golurk, Rookidee, Corvisquire, Corviknight, Igglybuff, Jigglypuff, Wigglytuff,
Fidough, Dachsbun, Starly, Staravia, Staraptor, Spoink, Grumpig, Squawkabilly, Crabrawler, Crabominable, Nacli,
Naclstack, Garganacl, Gulpin, Swalot, Zubat, Golbat, Crobat, Charcadet, Armarouge, Ceruledge, Maschiff, Mabosstiff,
Toxel, Toxtricity, Shroodle, Grafaiai, Zangoose, Seviper, Mime Jr., Mr. Mime, Galarian Mr. Mime, Mr. Rime, Foongus,
Amoonguss, Heatran, Volcanion, Cobalion, Terrakion, Virizion, Keldeo, Meloetta, Genesect, Hoopa, Marshadow, Meltan,
Melmetal, Darkrai, Latias, Latios, Kyogre, Groudon, Rayquaza, Magearna, Zeraora"""

CHAMPIONS = """Venusaur, Charizard, Blastoise, Beedrill, Pidgeot, Arbok, Pikachu, Raichu, Alolan Raichu, Clefable,
Ninetales, Alolan Ninetales, Wigglytuff, Vileplume, Persian, Alolan Persian, Arcanine, Hisuian Arcanine, Alakazam,
Machamp, Victreebel, Slowbro, Galarian Slowbro, Farfetch'd, Gengar, Kangaskhan, Starmie, Mr. Mime, Pinsir, Tauros,
Paldean Tauros, Gyarados, Ditto, Vaporeon, Jolteon, Flareon, Aerodactyl, Snorlax, Dragonite, Meganium, Typhlosion,
Hisuian Typhlosion, Feraligatr, Ariados, Ampharos, Azumarill, Politoed, Espeon, Umbreon, Slowking,
Galarian Slowking, Forretress, Steelix, Qwilfish, Scizor, Heracross, Skarmory, Houndoom, Tyranitar, Sceptile,
Blaziken, Swampert, Pelipper, Gardevoir, Sableye, Mawile, Aggron, Medicham, Manectric, Swalot, Sharpedo, Camerupt,
Torkoal, Altaria, Milotic, Castform, Banette, Chimecho, Absol, Glalie, Salamence, Metagross, Torterra, Infernape,
Empoleon, Staraptor, Luxray, Roserade, Rampardos, Bastiodon, Lopunny, Spiritomb, Garchomp, Lucario, Hippowdon,
Toxicroak, Abomasnow, Weavile, Rhyperior, Leafeon, Glaceon, Gliscor, Mamoswine, Gallade, Froslass, Rotom,
Serperior, Emboar, Samurott, Hisuian Samurott, Watchog, Liepard, Simisage, Simisear, Simipour, Musharna, Excadrill,
Audino, Conkeldurr, Scolipede, Whimsicott, Krookodile, Scrafty, Cofagrigus, Garbodor, Zoroark, Hisuian Zoroark,
Reuniclus, Vanilluxe, Emolga, Eelektross, Chandelure, Beartic, Stunfisk, Galarian Stunfisk, Golurk, Hydreigon,
Volcarona, Chesnaught, Delphox, Greninja, Diggersby, Talonflame, Vivillon, Pyroar, Eternal Flower Floette, Florges,
Gogoat, Pangoro, Furfrou, Meowstic, Aegislash, Aromatisse, Slurpuff, Malamar, Barbaracle, Dragalge, Clawitzer,
Heliolisk, Tyrantrum, Aurorus, Sylveon, Hawlucha, Dedenne, Goodra, Hisuian Goodra, Klefki, Trevenant, Gourgeist,
Avalugg, Hisuian Avalugg, Noivern, Decidueye, Hisuian Decidueye, Incineroar, Primarina, Toucannon, Crabominable,
Lycanroc, Toxapex, Mudsdale, Araquanid, Salazzle, Tsareena, Oranguru, Passimian, Golisopod, Mimikyu, Drampa,
Kommo-o, Rillaboom, Cinderace, Inteleon, Corviknight, Thievul, Flapple, Appletun, Sandaconda, Toxtricity,
Grapploct, Polteageist, Hatterene, Grimmsnarl, Perrserker, Sirfetch'd, Mr. Rime, Runerigus, Alcremie, Falinks,
Pincurchin, Indeedee, Morpeko, Dragapult, Wyrdeer, Kleavor, Basculegion, Sneasler, Overqwil, Meowscarada,
Skeledirge, Quaquaval, Pawmot, Maushold, Arboliva, Squawkabilly, Garganacl, Armarouge, Ceruledge, Bellibolt,
Mabosstiff, Scovillain, Espathra, Tinkaton, Palafin, Orthworm, Glimmora, Houndstone, Annihilape, Farigiraf,
Kingambit, Baxcalibur, Gholdengo, Sinistcha, Archaludon, Hydrapple"""

GAMES = {
    "sv": {
        "label": "Pokémon Scarlet / Violet (+ DLC)",
        "pokedexes": ["paldea", "kitakami", "blueberry"],
        "static": None,
        "version_groups": ["scarlet-violet", "the-teal-mask", "the-indigo-disk"],
        "default_level": 50,
        "has_tera": True, "has_abilities": True,
    },
    "za": {
        "label": "Pokémon Legends: Z-A (+ Mega Dimension)",
        "pokedexes": [],
        "static": ZA_LUMIOSE + ",\n" + ZA_HYPERSPACE,
        "version_groups": ["legends-za", "legends-z-a", "mega-dimension"],
        "default_level": 50,
        "has_tera": False, "has_abilities": False,
    },
    "champions": {
        "label": "Pokémon Champions",
        "pokedexes": [],
        "static": CHAMPIONS,
        "version_groups": ["champions", "pokemon-champions"],
        "default_level": 50,
        "has_tera": False, "has_abilities": True,
    },
}

_REGIONAL = {"Alolan": "alola", "Galarian": "galar", "Hisuian": "hisui", "Paldean": "paldea"}
_SPECIAL = {
    "paldean tauros": ("tauros", "tauros-paldea-combat-breed"),
    "eternal flower floette": ("floette", "floette-eternal"),
    "galarian darmanitan": ("darmanitan", "darmanitan-galar-standard"),
}


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("'", "").replace("’", "").replace(".", "").replace(":", "")
    s = s.replace("♀", "-f").replace("♂", "-m")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def parse_entry(display: str) -> dict:
    """'Alolan Raichu' -> {'display','species':'raichu','pokemon':'raichu-alola','form':'alola'}"""
    display = display.strip()
    low = display.lower()
    if low in _SPECIAL:
        sp, pk = _SPECIAL[low]
        return {"display": display, "species": sp, "pokemon": pk, "form": pk.split("-", 1)[1]}
    first, _, rest = display.partition(" ")
    if first in _REGIONAL and rest:
        sp = slugify(rest)
        form = _REGIONAL[first]
        return {"display": display, "species": sp, "pokemon": f"{sp}-{form}", "form": form}
    sp = slugify(display)
    return {"display": display, "species": sp, "pokemon": sp, "form": None}


def static_roster(game: str) -> list[dict]:
    txt = GAMES[game]["static"] or ""
    seen, out = set(), []
    for raw in txt.replace("\n", " ").split(","):
        if not raw.strip():
            continue
        e = parse_entry(raw)
        if e["pokemon"] not in seen:
            seen.add(e["pokemon"])
            out.append(e)
    return out
