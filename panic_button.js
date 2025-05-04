
var PanicButtonHandler = function() {
    this.render = function(model_state) {
        // Add panic button if it doesn't exist
        if (!document.getElementById("panic-button")) {
            console.log("Adding panic button");
            var button = document.createElement("button");
            button.id = "panic-button";
            button.innerHTML = "TRIGGER PANIC!";
            button.style.position = "fixed";
            button.style.top = "60px";
            button.style.right = "10px";
            button.style.backgroundColor = "red";
            button.style.color = "white";
            button.style.padding = "10px";
            button.style.fontWeight = "bold";
            button.style.border = "none";
            button.style.borderRadius = "5px";
            button.style.cursor = "pointer";
            button.style.zIndex = "1000";
            
            button.onclick = function() {
                console.log("Panic button clicked!");
                // Send a custom message through the websocket
                var message = {
                    "type": "panic_inject",
                    "method": "spatial"
                };
                ws.send(JSON.stringify(message));
            };
            
            document.body.appendChild(button);
            
            // Add a second button for random panic
            var randomButton = document.createElement("button");
            randomButton.id = "random-panic-button";
            randomButton.innerHTML = "RANDOM PANIC";
            randomButton.style.position = "fixed";
            randomButton.style.top = "110px";
            randomButton.style.right = "10px";
            randomButton.style.backgroundColor = "purple";
            randomButton.style.color = "white";
            randomButton.style.padding = "10px";
            randomButton.style.fontWeight = "bold";
            randomButton.style.border = "none";
            randomButton.style.borderRadius = "5px";
            randomButton.style.cursor = "pointer";
            randomButton.style.zIndex = "1000";
            
            randomButton.onclick = function() {
                console.log("Random panic button clicked!");
                // Send a custom message through the websocket
                var message = {
                    "type": "panic_inject",
                    "method": "random"
                };
                ws.send(JSON.stringify(message));
            };
            
            document.body.appendChild(randomButton);
        }
    };
};
            