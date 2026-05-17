# SHL AI Assessment Recommendation System

This project is an AI-powered recommendation system built for the SHL GenAI Assessment.

## Session Isolation Smoke Test

Run `session_isolation_smoke_test.py` to verify that session state stays isolated across multiple `session_id` values and that add/remove/confirm flows only affect the active session.

```powershell
& 'c:/Users/BIT/Desktop/SHL 26/.venv/Scripts/python.exe' 'c:/Users/BIT/Desktop/SHL 26/session_isolation_smoke_test.py'
```