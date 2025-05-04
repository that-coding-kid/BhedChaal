
var KeyboardHandler = function() {
    // Add event listener for keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        console.log("Key pressed: " + event.key);
        
        switch(event.key) {
            case 'p':
            case 'P':
                console.log("Panic key pressed!");
                // Send panic event
                var message = {
                    "type": "panic_inject",
                    "method": "spatial"
                };
                ws.send(JSON.stringify(message));
                break;
                
            case 'r':
            case 'R':
                console.log("Random panic key pressed!");
                // Send random panic event
                var message = {
                    "type": "panic_inject",
                    "method": "random"
                };
                ws.send(JSON.stringify(message));
                break;
        }
    });
    
    this.render = function(model_state) {
        // Add instructions for keyboard shortcuts if they don't exist
        if (!document.getElementById("keyboard-instructions")) {
            console.log("Adding keyboard instructions");
            var div = document.createElement("div");
            div.id = "keyboard-instructions";
            div.innerHTML = "<strong>Keyboard Shortcuts:</strong><br>" +
                           "Press 'P' to inject panic<br>" +
                           "Press 'R' to inject random panic";
            div.style.position = "fixed";
            div.style.bottom = "10px";
            div.style.right = "10px";
            div.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
            div.style.color = "white";
            div.style.padding = "10px";
            div.style.borderRadius = "5px";
            div.style.zIndex = "1000";
            
            document.body.appendChild(div);
        }
    };
};
            