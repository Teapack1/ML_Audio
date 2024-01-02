let ws;

function toggleButton(isStart) {
    const startButton = document.getElementById('startBtn');
    if (isStart) {
        startButton.innerText = "Start Recording";
        startButton.className = "start-button"; // CSS class for the start button
    } else {
        startButton.innerText = "Stop";
        startButton.className = "stop-button"; // CSS class for the stop button
    }
}

function startRecording() {
    // Initialize WebSocket connection only if it's not already established
    if (!ws || ws.readyState === WebSocket.CLOSED) {
        ws = new WebSocket("ws://localhost:8000/ws");

        ws.onopen = function(event) {
            ws.send("Start recording"); // Trigger the server to start the audio processing
        };

        ws.onmessage = function(event) {
            const serverMessage = event.data;
            document.getElementById('results').innerText = serverMessage;
            if (serverMessage.includes("Wake word detected")) {
                toggleButton(false); // Change to "Stop" button
            }
            if (serverMessage === "Restarting system...") {
                toggleButton(true); // Change back to "Start" button
                displayRestartButton();
            }
        };
    }
}

function displayRestartButton() {
    const restartBtn = document.createElement("button");
    restartBtn.innerText = "Restart";
    restartBtn.onclick = function() {
        location.reload(); // Reload the page to restart the process
    };
    document.getElementById('restart-button-container').appendChild(restartBtn);
}

document.getElementById('startBtn').addEventListener('click', startRecording);
