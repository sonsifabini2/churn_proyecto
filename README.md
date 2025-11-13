# Proyecto Churn Prediction

Este proyecto analiza el comportamiento de clientes y construye modelos predictivos de # churn (fuga de clientes) #  para un e-commerce usando datos en # PostgreSQL (Aiven.io) #. El foco es detectar a tiempo clientes con alta probabilidad de inactividad para habilitar acciones de retención.


## 📂 Estructura del Proyecto
proyecto_churn/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│ └── 01_exploracion_datos.ipynb
├── src/
│ ├── init.py
│ ├── conexion.py
│ ├── procesamiento.py
│ └── modelado.py
├── data/
│ ├── raw/
│ └── processed/
└── mlruns/

## 🎯 Objetivo de negocio
- **Reducir churn** anticipando clientes con alta probabilidad de inactividad, para activar campañas de retención (e.g., cupones, emails, remarketing).

- **Segmentar** por probabilidad de fuga y contribución (RFM) para priorizar acciones costo-efectivas (alto valor / alto riesgo primero).

- **Medir impacto:** utilizar recall alto en clase churn para no dejar ir clientes recuperables, sin perder de vista precision para optimizar presupuesto.

# Definición de churn 
- **Churn = Inactividad de 90 días** posteriores a un punto de corte (T0).
Por qué 90 días: equilibrio entre (a) ciclos de recompra típicos en e-commerce generalistas, (b) señal suficientemente fuerte para accionar retención, y (c) evitar falsos positivos de clientes con hábitos de recompra más espaciados.

- **Ventana histórica (features): 180 días previos** a T0.
Permite capturar frecuencia, gasto, ticket, variedad y señales estacionales sin cruzar el límite temporal (evitando leakage).


# Modelos y resultados 

# Modelo 1 - Ventana H=180, P=90
Se construyeron features RFM (recency, frequency, monetary), señales de comportamiento, variedad de consumo, actividad reciente, métricas de envío y pagos.
Se entrenaron Regresión Logística y Random Forest en pipelines con imputación y escalado.
Se evaluó con accuracy, precision, recall, F1, ROC-AUC, matrices de confusión y umbrales.

**Conclusión:** Sistema capaz de identificar clientes con alta probabilidad de churn basado en comportamiento real, manteniendo interpretabilidad (LR) y robustez (RF).


# Modelo 2 - Dataset RFM_CLIENTS_V2
- Dataset: `RFM_CLIENTS_V2` con variables RFM y agregaciones por cliente.
- Modelos: Regresión Logística, Decision Tree y Random Forest.
- Resultados (resumen): desempeño muy alto en test (accuracy y ROC-AUC ≈ 1.00).  
  - Decision Tree: “perfecto” en test → riesgo de overfitting.  
  - Random Forest: muy alto rendimiento, mayor estabilidad y robustez.  
  - Regresión Logística: casi perfecto y altamente interpretable.
  
**Conclusión:** Elegimos **Random Forest** como modelo productivo por su balance entre performance y estabilidad; mantenemos **Regresión Logística** como baseline interpretable. Recomendado validar con CV temporal y monitoreo en producción.


# Modelo 3 - Alternativo sin timestamps/recency

- H=180, P=90, target = is_inactive90.
- Exclusión explícita de variables basadas en recency y timestamps para evitar fuga de información.
- Variables usadas: frecuencia, gasto, ticket promedio, variedad, métricas por mes (todas previas a T0).

**Resultados**
- Logistic Regression: ROC-AUC 0.982, recall (churn) 0.964 → ideal para no dejar escapar clientes en riesgo.
- Random Forest: ROC-AUC 0.974, precision 0.890 → enfoque más conservador.

**Conclusión:** Alta capacidad discriminatoria sin información filtrada; sólido para estrategias reales de retención.













