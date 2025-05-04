
var FixedCanvasVisualization = function() {
    const width = 800;
    const height = 600;
    
    // Create a canvas element dynamically
    var canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.style.border = "1px solid black";
    
    // Add event listeners for keyboard controls
    document.addEventListener('keydown', function(event) {
        console.log("Key pressed:", event.key);
        
        if (event.key === 'p' || event.key === 'P') {
            console.log("Triggering panic!");
            // Create a custom message
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_panic": true
            };
            // Send it through the websocket
            ws.send(JSON.stringify(message));
        }
        
        if (event.key === 'r' || event.key === 'R') {
            console.log("Triggering random panic!");
            // Create a custom message
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_random_panic": true
            };
            // Send it through the websocket
            ws.send(JSON.stringify(message));
        }
    });
    
    // Create panic buttons
    function createPanicButtons() {
        if (document.getElementById("panic-button")) return;
        
        var panicBtn = document.createElement("button");
        panicBtn.id = "panic-button";
        panicBtn.innerHTML = "TRIGGER PANIC";
        panicBtn.style.position = "fixed";
        panicBtn.style.top = "100px";
        panicBtn.style.right = "20px";
        panicBtn.style.padding = "10px";
        panicBtn.style.backgroundColor = "red";
        panicBtn.style.color = "white";
        panicBtn.style.border = "none";
        panicBtn.style.fontWeight = "bold";
        panicBtn.style.cursor = "pointer";
        panicBtn.style.zIndex = "1000";
        
        panicBtn.onclick = function() {
            console.log("Panic button clicked");
            var message = {
                "type": "request_step",
                "step": 1,
                "trigger_panic": true
            };
            ws.send(JSON.stringify(message));
        };
        
        document.body.appendChild(panicBtn);
    }
    
    this.render = function(data) {
        // Make sure canvas is in the DOM
        if (!canvas.parentNode) {
            var elements = document.getElementById("elements");
            if (elements) {
                elements.appendChild(canvas);
            } else {
                document.body.appendChild(canvas);
            }
            // Create buttons when canvas is added
            createPanicButtons();
        }
        
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, width, height);
        
        // First, draw all the polygon elements (areas and roads)
        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            if (d.Shape === "polygon") {
                var points = d.Points;
                if (points && points.length > 0) {
                    ctx.beginPath();
                    ctx.moveTo(points[0][0], points[0][1]);
                    
                    for (var j = 1; j < points.length; j++) {
                        ctx.lineTo(points[j][0], points[j][1]);
                    }
                    
                    ctx.closePath();
                    ctx.strokeStyle = d.Color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    if (d.Filled === "true") {
                        ctx.fillStyle = d.Color;
                        ctx.fill();
                    }
                }
            }
        }
        
        // Draw all agents as static circles in a grid for debugging
        var agentCount = 0;
        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            if (d.Shape === "circle") {
                // Determine grid position
                var row = Math.floor(agentCount / 20);
                var col = agentCount % 20;
                
                // Calculate position in a grid layout
                var x = 50 + col * 35;
                var y = 50 + row * 35;
                
                // Draw the agent at the grid position
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, Math.PI * 2);
                ctx.fillStyle = d.Color;
                ctx.fill();
                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1;
                ctx.stroke();
                
                agentCount++;
            }
        }
        
        // Add text to canvas showing agent count
        ctx.fillStyle = "#000000";
        ctx.font = "20px Arial";
        ctx.fillText("Agents: " + agentCount, 10, 30);
        
        // Add instructions
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(10, height - 80, 380, 70);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "14px Arial";
        ctx.fillText("Press 'P' to trigger panic at a random location", 20, height - 55);
        ctx.fillText("Press 'R' to trigger random panic across agents", 20, height - 35);
        ctx.fillText("Blue = Normal agents, Red = Panicked agents", 20, height - 15);
    };
    
    this.reset = function() {
        // Nothing needed
    };
};
    