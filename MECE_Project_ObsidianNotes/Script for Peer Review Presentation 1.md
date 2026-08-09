	### **Slide 1: Problem Statement (0:00 – 0:45)**

**Visual:** _Title & Problem Statement_

"Hello everyone, I’m Dhanush Kumar Ramesh. Today, I’m presenting my research on the ' Yield-Aware Inverse Design of High-Speed Interconnects: A Physics-Constrained Generative Approach'

In our modern digital world, we are pushing data speeds to extreme limits—specifically 112 Gigabits per second. At these speeds, the margin for error in hardware design practically disappears. Even a microscopic drill error of just 10% during manufacturing can completely destroy a device's signal integrity.

Currently, the industry relies on a slow 'guess-and-check' method, running thousands of simulations just to ensure a design will work once it leaves the factory. My research asks: Can we use Artificial Intelligence to instantly generate designs that are not only high-performing but also resilient to these manufacturing errors?"

### **Slide 2: Visual Hook (0:45 – 1:05)**

**Visual:** _AI Chip and 3D Via Array images_

"To visualize this, think of these microscopic copper pathways, called 'vias,' as high-speed data highways. On the left, you see the complex chips that power our AI; on the right, the internal 3D structures that connect them. My goal is to ensure these highways stay open and efficient, even when the manufacturing process isn't perfect."

### **Slide 3: Architectural Framework (1:05 – 2:00)**

**Visual:** _The Flowchart & Methodology_

"To solve this, I’ve developed a **Tandem CVAE architecture**.

First, we use a **Generative Model** to solve the 'one-to-many' problem. This means for one desired performance goal, the AI can propose multiple valid 3D shapes.

Second, we have a **Forward Model**, which I’ve built using a Direct Sequence ResNet. This acts as our internal physics expert, instantly predicting performance without needing slow traditional simulators.

The 'secret sauce' for manufacturing yield is our **Jacobian Loss**. By mathematically penalizing sharp performance peaks, the AI learns to find 'stable plateaus.' This ensures that if a drill is slightly off-center during manufacturing, the device still performs exactly as intended. We are moving from designing for 'perfection' to designing for 'reality.'"

### **Slide 4: Value and Impact (2:00 – 2:35)**

**Visual:** _Manufacturability, Speed, and Next-Gen Icons_

"The impact of this work is a game-changer for hardware R&D.

We are reducing design cycles that normally take hours of human effort and simulation time down to just a few milliseconds of AI generation.

By creating designs that are 100% manufacturable and robust to tolerances, we can accelerate the rollout of next-generation 6G networks and AI hardware. Essentially, we are providing engineers with a tool that filters out 'bad' design options before they ever reach the factory floor."

### **Slide 5: Project Plan & Milestones (2:35 – 2:55)**

**Visual:** _Gantt Chart & Completed Milestones_

"Our roadmap is well underway. To date, I have completed the full data pipeline, processing over 18,000 differential pairs. I have also successfully trained the Forward Physics Model. Currently, I am implementing the Inverse Generative Model, with final benchmarking and performance validation against industry-standard simulators scheduled for completion by August."

### **Slide 6: Conclusion (2:55 – 3:00)**

**Visual:** _References_

"This research bridges the gap between deep learning and electromagnetic reality. Thank you for your time, and I look forward to your questions."