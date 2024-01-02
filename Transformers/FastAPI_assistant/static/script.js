let ws;

function startWebSocket() {
    ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = function(event) {
        const serverMessage = event.data;
        document.getElementById('results').innerText = serverMessage;
        if (serverMessage === "Restarting system...") {
            displayRestartButton();
        }
    };
}

function startRecording() {
    ws.send("Start recording"); // This message would trigger the server to start the audio processing
}

function displayRestartButton() {
    const restartBtn = document.createElement("button");
    restartBtn.innerText = "Restart";
    restartBtn.onclick = function() {
        location.reload(); // Reload the page to restart the process
    };
    document.body.appendChild(restartBtn);
}

document.getElementById('startBtn').addEventListener('click', startRecording);

window.onload = startWebSocket;
