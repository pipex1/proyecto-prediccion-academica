# Predicción de rendimiento académico

Aplicación web (Flask) que explica y compara dos modelos de predicción de aprobación,
**Regresión Logística** y **Red Neuronal Artificial**, entrenados con datos sintéticos.

## Ejecutar

```
pip install -r requirements.txt
python app.py
```

Abrir http://127.0.0.1:5000

- `/` explicación de los modelos y su uso en un sistema académico.
- `/practica` formulario para predecir con ambos modelos y comparar resultados individualmente o por lotes.

## Predicción por lotes

En `/practica`, carga un CSV con estas columnas:

| Columna | Valores |
|---|---|
| `asistencia` | Porcentaje entre 0 y 100 |
| `tareas_entregadas` | Entero entre 0 y 3 |
| `promedio_previo` | Promedio entre 0 y 5 |
| `aprobado` | Resultado real: 0 = reprueba, 1 = aprueba |

La columna `aprobado` es necesaria para comparar las predicciones con los resultados reales y calcular exactitud,
precisión, sensibilidad, F1 y la matriz de confusión de cada modelo. El botón de ejemplo descarga datos sintéticos
reservados para prueba. Los CSV pueden contener hasta 5.000 estudiantes y tener un tamaño máximo de 2 MB; la página
permite descargar las predicciones del lote completo.

## Reglas del curso simulado

- 32 horas de estudio: 16 clases de 2 horas.
- 3 actividades evaluativas con nota de 0 a 5; una actividad no entregada vale 0.0.
- Se aprueba con asistencia de 80% o más y promedio de 3.0 o más.
- Con asistencia mayor a 87% se suman 0.5 puntos al promedio (máximo 5.0).

Las constantes están al inicio de `app.py`. Los datos son simulados con fines educativos.
