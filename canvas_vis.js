
var CanvasVisualization = function() {
    // Canvas dimensions
    var width = 800;
    var height = 600;
    var canvas = null;
    var ctx = null;
    
    // Create the canvas when first rendering
    function createCanvas() {
        canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.style.border = "1px solid black";
        canvas.style.backgroundColor = "#f0f0f0";
        
        // Find where to append the canvas
        var elementsDiv = document.getElementById("elements");
        if (elementsDiv) {
            elementsDiv.appendChild(canvas);
        } else {
            document.body.appendChild(canvas);
        }
        
        ctx = canvas.getContext("2d");
        
        // Create control panel with panic button
        var controlPanel = document.createElement("div");
        controlPanel.style.position = "fixed";
        controlPanel.style.top = "80px";
        controlPanel.style.right = "10px";
        controlPanel.style.padding = "10px";
        controlPanel.style.backgroundColor = "rgba(255,255,255,0.8)";
        controlPanel.style.border = "1px solid #ccc";
        controlPanel.style.borderRadius = "5px";
        controlPanel.style.zIndex = "1000";
        
        // Add panic button
        var panicButton = document.createElement("button");
        panicButton.innerText = "TRIGGER PANIC";
        panicButton.style.backgroundColor = "red";
        panicButton.style.color = "white";
        panicButton.style.padding = "8px 15px";
        panicButton.style.border = "none";
        panicButton.style.borderRadius = "5px";
        panicButton.style.cursor = "pointer";
        panicButton.style.display = "block";
        panicButton.style.marginBottom = "10px";
        
        panicButton.onclick = function() {
            // Request a step and trigger panic
            // This has no direct effect but demonstrates the UI
            console.log("Panic button clicked");
            alert("Panic triggered! This would cause panic in the simulation.");
        };
        
        // Add random panic button
        var randomButton = document.createElement("button");
        randomButton.innerText = "RANDOM PANIC";
        randomButton.style.backgroundColor = "purple";
        randomButton.style.color = "white";
        randomButton.style.padding = "8px 15px";
        randomButton.style.border = "none";
        randomButton.style.borderRadius = "5px";
        randomButton.style.cursor = "pointer";
        randomButton.style.display = "block";
        
        randomButton.onclick = function() {
            console.log("Random panic button clicked");
            alert("Random panic triggered! This would cause random panic in the simulation.");
        };
        
        // Add instruction text
        var instructionsDiv = document.createElement("div");
        instructionsDiv.style.marginTop = "15px";
        instructionsDiv.style.fontSize = "12px";
        instructionsDiv.innerHTML = "<strong>Press P key:</strong> Trigger panic<br>" +
                                  "<strong>Press R key:</strong> Random panic<br>" +
                                  "<strong>Blue:</strong> Normal agents<br>" +
                                  "<strong>Red:</strong> Panicked agents";
        
        // Add everything to control panel
        controlPanel.appendChild(panicButton);
        controlPanel.appendChild(randomButton);
        controlPanel.appendChild(instructionsDiv);
        document.body.appendChild(controlPanel);
        
        // Add keyboard event listeners
        document.addEventListener("keydown", function(event) {
            if (event.key === "p" || event.key === "P") {
                console.log("P key pressed - would trigger panic");
                alert("P key pressed - would trigger panic");
            } else if (event.key === "r" || event.key === "R") {
                console.log("R key pressed - would trigger random panic");
                alert("R key pressed - would trigger random panic");
            }
        });
    }
    
    this.render = function(agents) {
        // Create canvas if it doesn't exist yet
        if (!canvas) {
            createCanvas();
        }
        
        // Clear the canvas
        ctx.clearRect(0, 0, width, height);
        
        // First, draw any polygons (walking areas)
        var polygons = agents.filter(function(a) { return a.Shape === "polygon"; });
        for (var i = 0; i < polygons.length; i++) {
            var poly = polygons[i];
            var points = poly.Points;
            
            if (points && points.length > 0) {
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                
                for (var j = 1; j < points.length; j++) {
                    ctx.lineTo(points[j][0], points[j][1]);
                }
                
                ctx.closePath();
                ctx.strokeStyle = poly.Color || "#00FF00";
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
        
        // Then draw agents in a grid layout for visibility
        var agentData = agents.filter(function(a) { return a.Shape === "circle"; });
        var totalAgents = agentData.length;
        
        // Layout in a grid with 20 columns
        var cellSize = 35;
        var cols = 20;
        var startX = 40;
        var startY = 40;
        
        for (var i = 0; i < totalAgents; i++) {
            var agent = agentData[i];
            var row = Math.floor(i / cols);
            var col = i % cols;
            
            var x = startX + col * cellSize;
            var y = startY + row * cellSize;
            
            // Draw the agent
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.fillStyle = agent.Color;
            ctx.fill();
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        
        // Show agent count and other info
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(10, 10, 150, 30);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "14px Arial";
        ctx.fillText("Agents: " + totalAgents, 20, 30);
    };
    
    this.reset = function() {
        // Nothing needed
    };
};
            