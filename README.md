# 🚗 EdgeMind — Closed-Loop Dynamic Autonomous Driving Digital Twin

> **Language-guided navigation with local AI, risk-aware A* pathfinding, and runtime safety validation.**

EdgeMind is a local AI-powered autonomous driving simulation that combines a Vision-Language Model (VLM), deterministic path planning, and a safety validation layer to enable intelligent navigation in dynamic environments.

It interprets natural-language driving instructions such as *"Drive to parking B and stay away from cones"*, extracts navigation constraints, generates an optimized route, and continuously monitors the environment to handle unexpected changes.

Built using Python, Streamlit, Ollama, and Qwen2.5-VL, EdgeMind demonstrates how language-based AI reasoning can work alongside deterministic planning and safety mechanisms on a local machine.

## ✨ Key Features

- **Natural Language Understanding:** Uses a locally running Qwen2.5-VL model to interpret driving instructions and extract destinations and obstacle-avoidance constraints.
- **Risk-Aware Path Planning:** Uses the A* algorithm to generate routes while applying penalties to cells near specified obstacles.
- **Runtime Safety Shield:** Independently validates planned paths before execution and prevents invalid or unsafe navigation.
- **Dynamic Environment:** Supports obstacles that appear, move, or disappear while the vehicle is navigating.
- **Closed-Loop Navigation:** Monitors environmental changes and replans routes when necessary.
- **Interactive World Editor:** Provides a 20×20 grid where users can add, move, and remove cars, cones, and barriers.
- **LLM vs A* Benchmarking:** Compares language-guided navigation against standard A* pathfinding using measurable performance metrics.
- **Local AI Execution:** Runs inference through Ollama without depending on paid cloud AI services.

## 🏗️ System Architecture

EdgeMind follows a hierarchical architecture in which the LLM handles high-level reasoning, while deterministic components handle navigation and safety.

```text
       Natural Language Instruction
                   |
                   v
          Instruction Gate
        (Driving Task Validation)
                   |
                   v
        VLM Constraint Extraction
          (Ollama + Qwen2.5-VL)
                   |
                   v
            A* Path Planner
         (Risk-Aware Navigation)
                   |
                   v
             Safety Shield
          (Path Validation)
                   |
                   v
         Navigation Execution
                   |
                   v
       Environment Monitoring
                   |
                   v
          Dynamic Replanning
```

### How It Works

1. **Instruction Validation:** A deterministic gate checks whether the input is a relevant driving instruction before invoking the LLM.
2. **Constraint Extraction:** The local VLM identifies the destination and obstacle types to avoid.
3. **Path Planning:** A* generates a route using obstacle information and additional penalties for soft-avoidance zones.
4. **Safety Validation:** The safety layer checks the generated path before the vehicle executes it.
5. **Closed-Loop Execution:** The environment is monitored during navigation. Changes trigger replanning when required.

**Design Principle:** The LLM acts as a high-level decision-maker rather than controlling every movement. The deterministic planner handles individual navigation steps, reducing unnecessary model calls and keeping execution efficient.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| AI Model | Qwen2.5-VL 7B |
| Local Model Runtime | Ollama |
| Pathfinding Algorithm | A* |
| Frontend | HTML, CSS, JavaScript |
| User Interface | Streamlit |
| Environment | Custom 20×20 Grid Simulation |

## 📊 Evaluation & Benchmarking

EdgeMind includes an LLM vs A* benchmark that evaluates both approaches on the same environment.

The benchmark measures whether language-guided obstacle constraints improve navigation compared with standard A* pathfinding.

**Key evaluation metrics:**

- Path length and number of navigation steps
- Number of hazard cells crossed
- Hazard Avoidance Rate
- Safety overrides
- Number of replanning operations
- Total LLM calls

### Example Benchmark Results

| Metric | LLM-Guided A* | A* Only |
|---|---:|---:|
| Navigation Steps | 24 | 22 |
| Hazard Cells Crossed | 0 | 2 |
| Arrival Status | Successful | Successful |

The LLM-guided approach avoided hazard cells at the cost of two additional navigation steps.

This demonstrates the trade-off between route efficiency and hazard avoidance.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd edgemind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the AI Model

```bash
ollama pull qwen2.5vl:7b
```

Make sure Ollama is running locally before launching the application.

### 4. Run EdgeMind

```bash
streamlit run drive_app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

## 🎮 Running the Demo

For a quick demonstration:

1. Select **Load Demo Scenario** to initialize the environment.
2. Enter a navigation instruction, such as *"Drive to parking B, stay away from cones."*
3. Start navigation and observe the generated route.
4. Monitor the live navigation panel for path information, hazard cells, and LLM calls.
5. Enable dynamic obstacles to observe how the system responds to environmental changes.
6. Run the **LLM vs A* Benchmark** to compare both navigation approaches.

## 📁 Project Structure

```text
edgemind/
│
├── drive_app.py              # Streamlit application and orchestration
├── drive_agent.py            # Navigation execution and replanning
├── drive_planner.py          # Instruction gate and Ollama VLM integration
├── drive_safety.py           # Runtime safety validation
├── index.html                # Interactive world editor
│
├── edgemind_editor/          # Streamlit editor component
│
├── vlmplan/
│   └── drive_world.py        # Driving world, obstacles, and A* planner
│
├── examples/
│   ├── evaluate.py           # Planner evaluation
│   └── scenes/               # Saved simulation scenes
│
├── docs/                     # Project documentation and figures
└── requirements.txt          # Python dependencies
```

## 🔒 Safety & Design Principles

EdgeMind separates AI reasoning from actual navigation execution.

- **Deterministic Validation:** Navigation instructions are checked before reaching the model.
- **Independent Safety Layer:** Every planned path is validated before execution.
- **Controlled LLM Usage:** The model handles high-level constraints rather than individual movement decisions.
- **No Silent Fallback:** The application does not silently switch to A*-only navigation when the required local LLM is unavailable.
- **Measurable Performance:** Navigation efficiency, hazard exposure, and safety interventions are recorded for evaluation.

## ⚠️ Current Limitations

- Soft-avoidance zones use a 3×3 neighborhood around specified obstacles.
- Navigation currently uses a 4-connected grid.
- Dynamic obstacle changes trigger a complete A* replanning operation.
- The current VLM primarily relies on textual scene information; full vision-based perception is still being developed.
- Some blocked and waiting states require improved recovery and timeout handling.

## 🔮 Future Scope

- Improved vision-based environmental perception.
- Incremental path planning using D* Lite.
- Enhanced safety validation and recovery mechanisms.
- More comprehensive LLM evaluation and performance analysis.
- Scene saving, loading, and replay functionality.
- Improved navigation visualization and debugging tools.

## 📄 License

This project is intended for educational and research purposes. See the [LICENSE](LICENSE) file for licensing details.

## 🙌 Acknowledgements

- [Ollama](https://ollama.com/) — Local AI model execution.
- [Streamlit](https://streamlit.io/) — Interactive application development.
- [Qwen](https://github.com/QwenLM) — Vision-Language Model.

---

**EdgeMind — Exploring Language-Guided Reasoning, Risk-Aware Planning, and Safety in Autonomous Navigation.**