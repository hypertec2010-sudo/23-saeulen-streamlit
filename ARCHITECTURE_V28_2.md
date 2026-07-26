# Architektur v28.2

```text
app.py
  -> modules/app_shell.py
      -> st.navigation / st.Page
          -> pages/*.py
              -> modules/page_runtime.py
                  -> legacy_app.py
                      -> Domain, Repositories, Storage und Fachmodule
```

`app.py` enthält nur noch Bootstrap, Zugriffsschutz und Navigation. Die Seiten setzen den benötigten Workspace und führen die weiterhin stabile Oberfläche über eine kontrollierte Runtime-Brücke aus. Damit ist die Navigation bereits nativ getrennt, während die nachfolgenden Releases die großen Renderbereiche schrittweise aus `legacy_app.py` in echte Page-Controller verschieben können.
