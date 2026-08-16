fix th@echo off
setlocal

set "BASE_URL=http://127.0.0.1:8000/generate"
set "PAYLOAD1={\"prompt\":\"Explain transformers in one paragraph.\",\"max_new_tokens\":64,\"stream\":false,\"temperature\":0.2,\"top_p\":0.9,\"top_k\":20}"
set "PAYLOAD2={\"prompt\":\"What is Kubernetes?\",\"max_new_tokens\":64,\"stream\":false,\"temperature\":0.2,\"top_p\":0.9,\"top_k\":20}"
set "PAYLOAD3={\"prompt\":\"What is PostgreSQL?\",\"max_new_tokens\":64,\"stream\":false,\"temperature\":0.2,\"top_p\":0.9,\"top_k\":20}"

echo Launching 3 requests in parallel...
start "" /b cmd /c curl.exe -s -o response1.json -X POST %BASE_URL% -H "Content-Type: application/json" -d "%PAYLOAD1%"
start "" /b cmd /c curl.exe -s -o response2.json -X POST %BASE_URL% -H "Content-Type: application/json" -d "%PAYLOAD2%"
start "" /b cmd /c curl.exe -s -o response3.json -X POST %BASE_URL% -H "Content-Type: application/json" -d "%PAYLOAD3%"

echo Waiting for the requests to finish...
timeout /t 20 /nobreak >nul

echo Done.
echo Check response1.json, response2.json, and response3.json.
echo Look at the server console for [scheduler] and [worker] logs.

endlocal
