# Federated Spark for Privacy-Preserving Diabetes Data Analytics

## Phase 1: Independent Development of Components

This project implements a federated learning pipeline with Apache Spark for diabetes data analytics, enabling collaborative model training across multiple institutions without sharing raw patient data.

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Java 8 or 11** (for Apache Spark)
3. **Git** (optional)

### Installation

```bash
# Clone or create project directory
mkdir diabetes_federated_spark
cd diabetes_federated_spark

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
diabetes_federated_spark/
│
├── data/
│   ├── raw/                    # Raw diabetes datasets
│   ├── processed/              # Preprocessed data
│   ├── streams/                # Streaming data
│   └── checkpoints/            # Spark checkpoints
│
├── models/                     # Saved models
├── results/                    # Experiment results
├── logs/                       # Log files
│
├── requirements.txt            # Python dependencies
├── data_preprocessing.py       # Data preprocessing script
├── spark_streaming_pipeline.py # Spark streaming implementation
├── federated_learning_model.py # PyTorch model definitions
├── fl_client.py               # Flower client implementation
├── fl_server.py               # Flower server implementation
├── run_phase1.py              # Complete pipeline runner
└── README.md                  # This file
```

---

## 🎯 Phase 1 Objectives

### 1. Spark Streaming Pipeline ✓
- Set up Apache Spark Structured Streaming
- Ingest and preprocess glucose + activity data
- Implement feature extraction
- Validate real-time data handling

### 2. Federated Learning Framework ✓
- Build basic FL setup using Flower
- Train models locally at simulated institutions
- Aggregate updates centrally (FedAvg)
- Validate global model convergence

---

## 🔧 Usage

### Option 1: Run Complete Pipeline (Recommended)

```bash
python run_phase1.py
```

This will execute all steps automatically:
1. Data preprocessing
2. Spark streaming pipeline
3. Federated learning training
4. Model comparison (centralized vs federated)

### Option 2: Run Components Individually

#### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

**What it does:**
- Loads diabetes data (or creates synthetic data)
- Handles missing values
- Extracts features (glucose trends, activity)
- Splits data for multiple institutions
- Saves data in streaming-ready format

#### Step 2: Spark Streaming Pipeline

```bash
python spark_streaming_pipeline.py
```

**What it does:**
- Simulates real-time data streams
- Processes streaming glucose + activity data
- Applies feature engineering
- Prepares data for ML training

**For multiple institutions:**
```bash
# Run for each institution
python spark_streaming_pipeline.py institution_1
python spark_streaming_pipeline.py institution_2
python spark_streaming_pipeline.py institution_3
```

#### Step 3: Federated Learning

**Option A: Simulation Mode (Easiest)**

```bash
python fl_server.py --simulate
```

This runs a complete FL simulation with all institutions in one process.

**Option B: Distributed Mode**

Terminal 1 (Server):
```bash
python fl_server.py
```

Terminal 2 (Client 1):
```bash
python fl_client.py institution_1
```

Terminal 3 (Client 2):
```bash
python fl_client.py institution_2
```

Terminal 4 (Client 3):
```bash
python fl_client.py institution_3
```

---

## 📊 Expected Output

### Data Preprocessing
```
Starting Data Preprocessing Pipeline
Loaded 10000 records
Preprocessed data shape: (10000, 10)
Institution 1: 3333 records
Institution 2: 3333 records
Institution 3: 3334 records
Preprocessing Complete!
```

### Spark Streaming
```
Starting Streaming Pipeline for institution_1
Streaming source created
Processing streaming data...

[institution_1] Statistics:
  Total Records: 3333
  Avg Glucose: 128.45 mg/dL
  Glucose Range: [42.3, 398.7]
  Hypo Events: 157

Pipeline completed for institution_1
```

### Federated Learning
```
Starting Federated Learning Simulation
Number of institutions: 3
Number of rounds: 10

Round 1:
[institution_1] Training complete
  Train Loss: 0.4523, Train Acc: 0.8234
  Val Acc: 0.8156, Val F1: 0.7891

Round 10:
Global Model Performance:
  Validation Accuracy: 0.8734
  F1 Score: 0.8512
  AUC: 0.9123

Federated Learning Complete!
```

---

## 🎓 Understanding the Code

### Data Preprocessing (`data_preprocessing.py`)

**Key Functions:**
- `preprocess_cgm_data()`: Cleans and normalizes glucose data
- `split_for_institutions()`: Simulates multiple healthcare institutions
- `save_for_streaming()`: Prepares data for streaming ingestion

**Features Created:**
- Glucose moving averages
- Glucose rate of change
- Time-based features (hour, day)
- Activity metrics (steps, heart rate)
- Risk indicators (hypoglycemia)

### Spark Streaming (`spark_streaming_pipeline.py`)

**Key Components:**
- `SparkStreamingPipeline`: Main pipeline class
- `create_streaming_source()`: Reads streaming JSON data
- `preprocess_stream()`: Real-time feature engineering
- `write_to_memory()`: Stores processed data for querying

**Streaming Features:**
- Real-time glucose monitoring
- Missing value imputation
- Feature normalization
- Window-based aggregations

### Federated Learning (`federated_learning_model.py`, `fl_client.py`, `fl_server.py`)

**Model Architecture:**
```
Input(10) -> Dense(64) -> BatchNorm -> ReLU -> Dropout
         -> Dense(32) -> BatchNorm -> ReLU -> Dropout
         -> Dense(16) -> BatchNorm -> ReLU -> Dropout
         -> Output(2)  [No Hypo Risk | Hypo Risk]
```

**Federated Process:**
1. **Local Training**: Each institution trains on local data
2. **Parameter Upload**: Send model updates to server
3. **Aggregation**: Server averages parameters (FedAvg)
4. **Global Update**: Distribute global model to clients
5. **Repeat**: Multiple rounds until convergence

**Privacy Preservation:**
- Raw data never leaves institutions
- Only model parameters are shared
- Secure aggregation (planned for Phase 2)

---

## 📈 Performance Metrics

The pipeline evaluates models using:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

**Target**: Hypoglycemia risk prediction (glucose < 70 mg/dL)

---

## 🔍 Verification & Validation

### Testing Spark Streaming

```python
# Check if streaming data is processed correctly
from spark_streaming_pipeline import SparkStreamingPipeline

pipeline = SparkStreamingPipeline("test_institution", "data/streams/test_input", "data/checkpoints/test")
final_df = pipeline.run_pipeline(duration_seconds=10)

# Verify processed records
print(f"Processed records: {final_df.count()}")
print(f"Schema: {final_df.schema}")
```

### Testing Federated Learning

```python
# Verify model convergence
import json

with open('results/fl_history_TIMESTAMP.json', 'r') as f:
    history = json.load(f)

print(f"Initial Accuracy: {history['val_accuracy'][0]:.4f}")
print(f"Final Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"Improvement: {(history['val_accuracy'][-1] - history['val_accuracy'][0]):.4f}")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Spark Not Starting**
```
Error: Java not found
```
**Solution**: Install Java 8 or 11 and set JAVA_HOME

**2. Memory Errors**
```
OutOfMemoryError: Java heap space
```
**Solution**: Increase Spark memory in `spark_streaming_pipeline.py`:
```python
.config("spark.driver.memory", "8g")
```

**3. Flower Connection Issues**
```
Error: Cannot connect to server
```
**Solution**: 
- Ensure server is running first
- Check firewall settings
- Use `localhost` instead of `0.0.0.0`

**4. Missing Data**
```
FileNotFoundError: data not found
```
**Solution**: Run `data_preprocessing.py` first, or the script will create synthetic data

---

## 📚 Datasets

### Option 1: Use Provided Datasets
- DiabetesData: https://www.kaggle.com/datasets/beyzacinar22/diadata
- OhioT1DM: https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

Download and place in `data/raw/`

### Option 2: Use Synthetic Data
The pipeline automatically generates synthetic data if no real data is found.

---

## 🎯 Success Criteria for Phase 1

- ✅ Spark streaming pipeline processes data in real-time
- ✅ Feature extraction handles missing values correctly
- ✅ Federated learning framework trains models locally
- ✅ FedAvg aggregation produces global model
- ✅ Global model converges and shows improvement
- ✅ Performance comparable to centralized baseline

---

## 🚀 Next Steps: Phase 2

Phase 2 will focus on:

1. **Integration**: Connect Spark with FL framework
2. **Privacy Enhancement**: 
   - Implement Secure Aggregation
   - Add Differential Privacy
3. **End-to-End Testing**: Complete pipeline validation
4. **Privacy-Utility Trade-offs**: Evaluate accuracy vs privacy

---

## 📖 References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. Apache Spark Structured Streaming Documentation
3. Flower: A Friendly Federated Learning Framework
4. Kairouz et al., "Advances and Open Problems in Federated Learning" (2021)

---

## 👥 Contributors

- KONIDENA SWAPNA (2022BCS0229)
- YESHWANTH I (2022BCD0055)
- YOGESH KUMAR SAINI (2022BCD0052)
- PAMBALA SANDEEP KUMAR (2022BCS0076)

**Guided By**: Dr. SHAJULIN BENEDICT

**Institution**: Indian Institute of Information Technology Kottayam

---

## 📝 License

This project is developed for academic purposes at IIIT Kottayam.

---

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in `logs/` directory
3. Contact project team members
4. Refer to official documentation:
   - Apache Spark: https://spark.apache.org/docs/latest/
   - Flower FL: https://flower.dev/docs/

---

**Last Updated**: October 2025
**Version**: 1.0 (Phase 1)
