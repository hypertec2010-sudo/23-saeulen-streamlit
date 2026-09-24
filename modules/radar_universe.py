"""Versioned radar discovery lists; not a complete or continuously verified market universe.

Only documented symbol corrections are automatic. Acquired companies are NEVER
silently mapped to their buyer. Actual provider failures remain scan diagnostics.
"""
from __future__ import annotations
import hashlib
import json
import re

CATALOG_VERSION = "2026-09-21-r1"
REVIEWED_AT = "2026-09-21"
ALIASES = {"SQ": "XYZ", "JFROG": "FROG"}
INACTIVE = {
    "JNPR": "Juniper-Uebernahme abgeschlossen; alte Notierung entfernt (2025-07-02).",
    "MAXR": "Maxar privatisiert; alte Notierung entfernt (2023-05-03).",
    "AJRD": "Aerojet Rocketdyne uebernommen; alte Notierung entfernt (2023-07-28).",
}
# Evidence belongs with the catalogue, not in every result card.
SOURCES = {
    "SQ": "https://investors.block.xyz/investor-news/news-details/2025/Block-Announces-Ticker-Symbol-Change-to-XYZ-To-Report-Fourth-Quarter-Results/default.aspx",
    "JFROG": "https://investors.jfrog.com/financials/quarterly-results/default.aspx",
    "JNPR": "https://www.hpe.com/us/en/newsroom/press-release/2025/07/hewlett-packard-enterprise-closes-acquisition-of-juniper-networks-to-offer-industry-leading-comprehensive-cloud-native-ai-driven-portfolio.html",
    "MAXR": "https://maxar.com/press-releases/u-s-private-equity-firm-advent-international-and-bci-complete-acquisition-of-maxar-technologies",
    "AJRD": "https://www.l3harris.com/newsroom/press-release/2023/07/l3harris-completes-aerojet-rocketdyne-acquisition",
}


def normalize_entries(entries):
    """Explicit aliases, delist exclusions and stable deduplication, no network."""
    symbols, notes, seen = [], [], set()
    for raw in entries:
        symbol = str(raw or "").strip().upper()
        if not symbol:
            continue
        if symbol in INACTIVE:
            notes.append({"input": symbol, "action": "removed", "detail": INACTIVE[symbol]})
            continue
        mapped = ALIASES.get(symbol, symbol)
        if mapped != symbol:
            notes.append({"input": symbol, "action": "alias", "detail": mapped})
        if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,23}", mapped):
            notes.append({"input": symbol, "action": "invalid", "detail": "Kein eindeutiger Ticker; bitte Symbol pruefen."})
            continue
        if mapped in seen:
            notes.append({"input": symbol, "action": "duplicate", "detail": mapped})
            continue
        seen.add(mapped)
        symbols.append(mapped)
    return symbols, notes


def get_universe_map():
    return {name: (normalize_entries(spec[0])[0], spec[1], spec[2])
            for name, spec in _seed_universes().items()}


def universe_notes(name):
    spec = _seed_universes().get(name)
    return normalize_entries(spec[0])[1] if spec else []


def universe_digest(symbols):
    # Input order is not an analytical input. Display limit is intentionally absent.
    return hashlib.sha256(json.dumps(sorted(set(symbols)), separators=(",", ":")).encode()).hexdigest()[:24]


def _seed_universes():
    us_tech_universe = [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "IBM",
        "QCOM", "TXN", "MU", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "INTC", "PANW",
        "CRWD", "SNPS", "CDNS", "ANET", "PLTR", "NOW", "ADSK", "TEAM", "ROP", "DELL",
        "HPQ", "WDAY", "DDOG", "NET", "MDB", "ZS", "OKTA", "HUBS", "SHOP", "SQ",
        "UBER", "ABNB", "META", "GOOGL", "AMZN", "NFLX", "TSM", "ASML", "ARM", "SMCI",
        "APH", "FTNT", "MCHP", "NXPI", "MRVL", "ON", "STM", "MPWR", "GFS", "WDC",
        "STX", "NTAP", "DOCU", "SNOW", "FICO", "TTD", "PINS", "SAP", "PATH", "ESTC",
        "DT", "APP", "RBLX", "GEN", "AKAM", "ZI", "BILL", "PAYC", "TYL", "MANH",
        "CYBR", "S", "IOT", "PCOR", "GWRE", "AFRM", "DOCN", "WK", "CFLT", "ENPH",
        "SEDG", "GLW", "JBL", "FSLR", "COHR", "CIEN", "JNPR", "FFIV", "TER", "ENTG"
    ]
    us_basis_universe = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "LLY", "V",
        "XOM", "UNH", "AVGO", "MA", "COST", "WMT", "JNJ", "PG", "HD", "ABBV",
        "BAC", "KO", "MRK", "PEP", "CVX", "ADBE", "CRM", "NFLX", "AMD", "ORCL",
        "LIN", "TMO", "MCD", "GE", "CAT", "AMAT", "GS", "AXP", "NOW", "PM"
    ]
    europa_quality_universe = [
        "SAP", "ASML", "NESN.SW", "NOVO-B.CO", "MC.PA", "SU.PA", "AIR.PA", "SIE.DE", "DTE.DE", "ALV.DE",
        "MUV2.DE", "RMS.PA", "OR.PA", "SAN.PA", "BN.PA", "DG.PA", "EL.PA", "SAF.PA", "CS.PA", "ULVR.L",
        "AZN.L", "SHEL.L", "REL.L", "LSEG.L", "DGE.L", "GSK.L", "ABBN.SW", "ROG.SW", "SIKA.SW", "UHR.SW",
        "NOVN.SW", "ZURN.SW", "CFR.SW", "ADYEN.AS", "WKL.AS", "PRX.AS", "HEIA.AS", "CAP.PA", "DSY.PA", "KER.PA",
        "RACE.MI", "MONC.MI", "UCG.MI", "ISP.MI", "PRY.MI", "ENEL.MI", "IBE.MC", "ITX.MC", "FER.MC", "AMS.MC",
        "HEI.DE", "IFX.DE", "DB1.DE", "RHM.DE", "RI.PA", "HO.PA", "AI.PA", "KER.PA", "LONN.SW", "HOLN.SW"
    ]
    europa_small_mid_quality_universe = [
        "NEM.DE", "BC8.DE", "COK.DE", "EVD.DE", "AFX.DE", "SIX2.DE", "FPE3.DE", "KRN.DE", "GXI.DE", "JUN3.DE",
        "G24.DE", "HAG.DE", "EVT.DE", "R3NK.DE", "PNE3.DE", "VAR1.DE", "VOS.DE", "WAF.DE", "DEQ.DE", "NDA.DE",
        "SOI.PA", "VIRP.PA", "SESL.PA", "IPS.PA", "EKI.PA", "RCO.PA", "UBI.PA", "EDEN.PA", "RXL.PA", "GET.PA",
        "IMCD.AS", "ASM.AS", "BESI.AS", "ASRNL.AS", "RAND.AS", "WKL.AS", "AD.AS", "FAGR.BR", "ACKB.BR", "SOF.BR",
        "LIFCO-B.ST", "ADDT-B.ST", "THULE.ST", "MIPS.ST", "NIBE-B.ST", "INDU-C.ST", "VITR.ST", "AAK.ST", "ALFA.ST", "SHB-A.ST",
        "DEMANT.CO", "GN.CO", "BAVA.CO", "CHEMM.CO", "NETC.CO", "RATO-B.ST", "SALM.OL", "TOM.OL", "KCR.HE", "VALMT.HE",
        "HUH1V.HE", "KEMIRA.HE", "METSB.HE", "TEL2-B.ST", "INDT.ST", "DIA.MI", "ERG.MI", "BFF.MI", "IP.MI", "REY.MI",
        "AMP.MC", "VID.MC", "LOG.MC", "CLNX.MC", "ANA.MC", "AUTO.MC", "TLGO.MC", "WISE.L", "LGEN.L", "WEIR.L"
    ]
    semiconductor_universe = [
        "NVDA", "AVGO", "AMD", "QCOM", "TXN", "MU", "ADI", "AMAT", "LRCX", "KLAC",
        "INTC", "MCHP", "NXPI", "MRVL", "ON", "MPWR", "GFS", "SWKS", "QRVO", "TER",
        "TSM", "ASML", "ARM", "STM", "ENTG", "COHR", "ONTO", "LSCC", "ALGM", "SLAB",
        "CRUS", "AMKR", "FORM", "IPGP", "NVMI", "ACLS", "POWI", "WOLF", "MTSI", "RMBS",
        "CAMT", "SITM", "ASX", "ASM.AS", "BESI.AS"
    ]
    us_small_mid_caps_universe = [
        "APP", "AFRM", "BILL", "CFLT", "CRSP", "CYBR", "DOCN", "DUOL", "ESTC", "FIVN",
        "FROG", "GLBE", "GWRE", "IOT", "JFROG", "MDB", "MGNI", "NET", "OKTA", "PCOR",
        "PLTR", "RBLX", "S", "SE", "SNOW", "SOFI", "U", "WIX", "ZI", "ZM",
        "RKLB", "IONQ", "ASTS", "CELH", "ELF", "ONON", "CAVA", "HIMS", "NU", "PINS",
        "DASH", "TTD", "ROKU", "ETSY", "CHWY", "DKNG", "HOOD", "ABNB", "UBER", "LYFT",
        "FSLY", "TASK", "COUR", "ASAN", "MNDY", "BROS", "CVNA", "CROX", "ACLS", "ALGM",
        "LSCC", "FORM", "SITM", "POWI", "WOLF", "COHR", "IPGP", "CAMT", "MTSI", "NVMI",
        "INSM", "AXSM", "EXAS", "HALO", "SRPT", "ALKS", "MEDP", "RXRX", "TWST", "NTLA"
    ]
    space_stocks_universe = [
        "RKLB", "ASTS", "LUNR", "RDW", "SPIR", "PL", "BKSY", "SATL", "IRDM", "VSAT",
        "GSAT", "MAXR", "BA", "LMT", "NOC", "RTX", "GD", "TDY", "HEI", "AJRD"
    ]
    quantum_computing_universe = [
        "IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "IBM", "GOOGL", "MSFT", "AMZN", "HON",
        "NVDA", "INTC", "FORM", "TER"
    ]
    software_universe = [
        # Large Cap / Plattform-Software
        "MSFT", "ORCL", "SAP", "CRM", "NOW", "ADBE", "INTU", "ADSK", "FICO", "TYL",
        # Cloud, Data, Observability, DevTools
        "SNOW", "MDB", "DDOG", "NET", "ESTC", "DT", "CFLT", "DOCN", "FROG", "GTLB",
        # Cybersecurity
        "PANW", "CRWD", "FTNT", "ZS", "OKTA", "CYBR", "S", "TENB", "VRNS", "QLYS",
        # SaaS / Business Applications
        "WDAY", "TEAM", "HUBS", "SHOP", "MNDY", "BILL", "PAYC", "PCOR", "GWRE", "DOCU",
        # AI-/App-/Data-nahe Software und Plattformen
        "PLTR", "APP", "TTD", "PATH", "U", "RBLX", "AFRM", "DUOL", "IOT", "MANH"
    ]
    emerging_markets_universe = [
        "EEM", "IEMG", "VWO", "KWEB", "MCHI", "FXI", "INDA", "EWZ", "EWT", "EWY",
        "EWW", "EIDO", "TUR", "EPOL", "ARGT", "NU", "MELI", "TSM", "BABA", "PDD",
        "JD", "BIDU", "SE", "GRAB", "TCOM", "NIO", "LI", "XPEV", "VALE", "PBR"
    ]
    return {
        "US Tech": (us_tech_universe, "US Tech Fokus", "Breites Tech- und Plattformuniversum mit rund 95 vordefinierten Werten."),
        "US Basisliste": (us_basis_universe, "US Basisliste", "Große US-Standardwerte als breiter Startscreen für neue Ideen."),
        "Europa Qualität & Leader": (europa_quality_universe, "Europa Qualität & Leader", "Breitere Europa-Liste mit Qualitätswerten, Large Caps und führenden Marktpositionen."),
        "Halbleiter": (semiconductor_universe, "Halbleiter", "Breite Halbleiterliste mit Designern, Ausrüstern, Foundries und Spezialwerten."),
        "US Small & Mid Caps": (us_small_mid_caps_universe, "US Small & Mid Caps", "Breiteres US-Universum aus Small- und Mid-Caps mit Fokus auf Liquidität, Wachstum und frühere Radar-Chancen."),
        "Europa Small & Mid Caps Qualität": (europa_small_mid_quality_universe, "Europa Small & Mid Caps Qualität", "Breiteres Europa-Universum aus Small- und Mid-Caps mit Qualitäts- und Leader-Fokus."),
        "Space Aktien": (space_stocks_universe, "Space Aktien", "Raumfahrt-, Satelliten- und Aerospace-nahe Titel; spekulativeres Themenuniversum mit hoher Volatilität."),
        "Quantencomputer": (quantum_computing_universe, "Quantencomputer", "Quantum-Computing-Pure-Plays und große Technologieanbieter mit Quantum-Exposure; stark thematisch und teils volatil."),
        "Software": (software_universe, "Software", "Software-, SaaS-, Cloud-, Cybersecurity- und Datenplattform-Werte als eigener Radar-Schwerpunkt."),
        "Emerging Markets": (emerging_markets_universe, "Emerging Markets", "EM-ETFs und große liquide Emerging-Markets-Werte als Makro-/Länder- und Wachstumsscreen."),
    }
