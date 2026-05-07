# ABGABE-CHECKLISTE

Stand: 2026-05-07 (Europe/Zurich)

## 1. Pflicht-Links

- GitHub Repo URL: https://github.com/Scampoloni/financial-market-predictor
- Frontend URL (Deployment): https://financial-market-predictorr.streamlit.app/
- Backend URL (separat): Nicht anwendbar (Streamlit-App, kein getrennt deploytes Spring-Backend in diesem Repo)
- Postman Documentation URL: Nicht anwendbar (keine REST-API/Postman-Artefakte in diesem Repo)
- Roadmap-History URL: Nicht vorhanden (optional manuell ergänzen, falls extern geführt)
- Iterations/Labels Chart URL: Nicht vorhanden (optional manuell ergänzen, falls extern geführt)
- SonarCloud URL: Nicht vorhanden (optional manuell ergänzen, falls aktiviert)

## 2. Verifikation (technisch)

- `pytest tests/ -q`: Erfolgreich, `8 passed` (lokal am 2026-05-07)
- Hinweis: 1 Warnung zu `.pytest_cache` (Windows Permission), ohne Einfluss auf Testresultat
- `backend/target/site/jacoco`: Nicht vorhanden (N/A, da dieses Repo kein Maven/Spring-Backend enthÃ¤lt)
- `backend/target/surefire-reports`: Nicht vorhanden (N/A, da dieses Repo kein Maven/Spring-Backend enthÃ¤lt)

## 3. Repo-Hygiene

- `.env` ist in `.gitignore` enthalten
- `.env` ist **nicht** getrackt (`git ls-files .env .env.example` listet nur `.env.example`)
- `.env.example` vorhanden und enthält nur Platzhalter
- Secrets-Scan über getrackte Dateien: keine harten Keys/Passwörter gefunden

## 4. Abgabe-Hinweise (nicht committen)

- Login/Passwort je Rolle separat in Moodle abgeben, nicht im Repository committen
- Dozierende als Collaborators im Repo hinzufügen: `jasminh`, `bkuehnis`
- Falls GitHub Project verwendet wird: Dozierende auch dort berechtigen

## 5. Manuelle Final-Checks vor Upload

- Deployment-Link im Browser öffnen und Kern-Flow live testen
- README-Links anklicken (Screenshots, Doku, Runbook)
- Sichtbarkeit des Repos (public oder korrekt freigegeben) bestätigen
