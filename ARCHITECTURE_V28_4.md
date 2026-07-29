# Architektur v28.4 – automatisierte Quality Gates

## Testpyramide

```text
GitHub Actions
├── Deployment-Verifikation
├── deterministische Regressionstests
├── Pytest-Unit-/Integrationstests
└── Streamlit AppTest
```

## Trennung der Refresh-Logik

```text
Streamlit Fragment
├── liest Session State
├── zeigt Countdown
└── löst st.rerun aus

modules/live_refresh_policy.py
├── baut Cache- und Schedule-Keys
├── berechnet Fälligkeit
└── verhindert doppelte Trigger
```

Die Policy ist frei von Streamlit und Netzwerkzugriffen. Dadurch kann das
Auto-Refresh-Verhalten mit festen Zeitpunkten reproduzierbar getestet werden.

## CI-Sicherheit

Der Workflow verwendet keine produktiven Streamlit- oder Supabase-Secrets.
Der echte App-Einstieg wird ausschließlich im unauthentifizierten Zustand
geprüft. Speicher- und Journaltests verwenden temporäre lokale Verzeichnisse.
