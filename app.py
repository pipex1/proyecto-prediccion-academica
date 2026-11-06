import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HORAS_ESTUDIO = 32
HORAS_POR_CLASE = 2
TOTAL_CLASES = HORAS_ESTUDIO // HORAS_POR_CLASE  # 16
TOTAL_ACTIVIDADES = 3  # una actividad evaluativa por entrega
NOTA_MIN, NOTA_MAX = 0.0, 5.0
FEATURES = ["horas_estudio", "asistencia", "tareas_entregadas", "promedio_previo"]
INPUT_FEATURES = ["asistencia", "tareas_entregadas", "promedio_previo"]
ASISTENCIA_MIN = 80.0  # % mínimo para aprobar
PROMEDIO_MIN = 3.0
ASISTENCIA_PREMIO = 87.0  # % por encima del cual se premia la asistencia
PUNTOS_PREMIO = 0.5
TARGET = "aprobado"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
MAX_FILAS_LOTE = 5000


def aplicar_premio(asistencia, promedio):
    """Suma 0.5 al promedio si la asistencia supera el 95% (máximo 5.0)."""
    premio = np.where(asistencia > ASISTENCIA_PREMIO, PUNTOS_PREMIO, 0.0)
    return np.minimum(promedio + premio, NOTA_MAX).round(2)


def generar_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    clases = rng.binomial(TOTAL_CLASES, 0.85, n)
    entregadas = rng.random((n, TOTAL_ACTIVIDADES)) < 0.85
    notas = np.clip(rng.normal(3.6, 1.0, (n, TOTAL_ACTIVIDADES)), NOTA_MIN, NOTA_MAX)
    notas = np.where(entregadas, notas, 0.0)  # sin entrega la nota es 0.0
    tareas = entregadas.sum(axis=1)
    asistencia = (clases / TOTAL_CLASES * 100).round(1)
    promedio = aplicar_premio(asistencia, notas.mean(axis=1))
    df = pd.DataFrame({
        "horas_estudio": HORAS_ESTUDIO,
        "asistencia": asistencia,
        "tareas_entregadas": tareas,
        "promedio_previo": promedio,
    })
    # Regla académica: se aprueba con asistencia >= 80% y promedio >= 3.0
    df[TARGET] = ((df.asistencia >= ASISTENCIA_MIN) & (df.promedio_previo >= PROMEDIO_MIN)).astype(int)
    return df


DATA = generar_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    DATA[FEATURES], DATA[TARGET], test_size=0.25, random_state=42, stratify=DATA[TARGET])

MODELOS = {
    # grado 3 permite a la regresión aprender la regla de dos umbrales (asistencia y promedio)
    "Regresión Logística": make_pipeline(
        StandardScaler(), PolynomialFeatures(3), StandardScaler(),
        LogisticRegression(C=1000, max_iter=20000)),
    "Red Neuronal Artificial": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42)),
}
for m in MODELOS.values():
    m.fit(X_train, y_train)

EXACTITUD = {n: round(float(m.score(X_test, y_test)) * 100, 1) for n, m in MODELOS.items()}


def contexto():
    return dict(
        horas_estudio=HORAS_ESTUDIO,
        horas_por_clase=HORAS_POR_CLASE, total_clases=TOTAL_CLASES,
        total_actividades=TOTAL_ACTIVIDADES,
        nota_min=NOTA_MIN, nota_max=NOTA_MAX, promedio_min=PROMEDIO_MIN,
        asistencia_premio=ASISTENCIA_PREMIO, puntos_premio=PUNTOS_PREMIO,
        asistencia_min=ASISTENCIA_MIN,
        exactitud=EXACTITUD, n_prueba=len(X_test), n_total=len(DATA))


@app.route("/")
def index():
    return render_template("index.html", teoria=True, **contexto())


@app.route("/practica")
def practica():
    return render_template("index.html", teoria=False, **contexto())


def calcular_entradas(datos):
    """Convierte clases asistidas y actividades en asistencia %, tareas y promedio."""
    clases = int(datos["clases_asistidas"])
    if not 0 <= clases <= TOTAL_CLASES:
        raise ValueError("clases fuera de rango")
    actividades = datos["actividades"]
    if len(actividades) != TOTAL_ACTIVIDADES:
        raise ValueError("número de actividades incorrecto")
    notas = []
    entregadas = 0
    for a in actividades:
        if a["entregada"]:
            nota = float(a["nota"])
            if not NOTA_MIN <= nota <= NOTA_MAX:
                raise ValueError("nota fuera de rango")
            entregadas += 1
        else:
            nota = 0.0  # sin entrega la nota es 0.0
        notas.append(nota)
    asistencia = round(clases / TOTAL_CLASES * 100, 1)
    promedio_base = round(sum(notas) / TOTAL_ACTIVIDADES, 2)
    promedio = float(aplicar_premio(asistencia, promedio_base))
    return {
        "asistencia": asistencia,
        "tareas_entregadas": entregadas,
        "promedio_base": promedio_base,
        "premio_asistencia": round(promedio - promedio_base, 2),
        "promedio_previo": promedio,
    }


@app.post("/api/individual")
def individual():
    datos = request.get_json(force=True, silent=True) or {}
    try:
        valores = calcular_entradas(datos)
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Completa clases asistidas y la nota (0–5) de cada actividad entregada."), 400
    fila = pd.DataFrame([{"horas_estudio": HORAS_ESTUDIO, **{f: valores[f] for f in INPUT_FEATURES}}],
                        columns=FEATURES)
    resultado = {"_entradas": valores}
    for nombre, modelo in MODELOS.items():
        prob = float(modelo.predict_proba(fila)[0, 1])
        resultado[nombre] = {"prediccion": "Aprueba" if prob >= 0.5 else "Reprueba",
                             "probabilidad_aprobar": round(prob, 4)}
    return jsonify(resultado)


@app.get("/descargar-ejemplo.csv")
def descargar_ejemplo():
    ejemplo = X_test[INPUT_FEATURES].copy()
    ejemplo[TARGET] = y_test.to_numpy()
    contenido = ejemplo.to_csv(index=False)
    return Response(
        contenido,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=ejemplo_prediccion_lote.csv"},
    )


@app.post("/api/lote")
def lote():
    archivo = request.files.get("archivo")
    if archivo is None or not archivo.filename:
        return jsonify(error="Selecciona un archivo CSV."), 400
    try:
        datos = pd.read_csv(archivo)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return jsonify(error="No se pudo leer el archivo. Verifica que sea un CSV válido."), 400

    columnas = [*INPUT_FEATURES, TARGET]
    faltantes = [columna for columna in columnas if columna not in datos.columns]
    if faltantes:
        return jsonify(error=f"Faltan columnas requeridas: {', '.join(faltantes)}."), 400
    if datos.empty:
        return jsonify(error="El archivo no contiene estudiantes."), 400
    if len(datos) > MAX_FILAS_LOTE:
        return jsonify(error=f"El archivo no puede superar {MAX_FILAS_LOTE} estudiantes."), 400

    datos = datos[columnas].apply(pd.to_numeric, errors="coerce")
    if datos.isna().any().any() or not np.isfinite(datos.to_numpy()).all():
        return jsonify(error="Todas las columnas deben contener valores numéricos válidos."), 400
    if not datos["asistencia"].between(0, 100).all():
        return jsonify(error="La asistencia debe estar entre 0 y 100."), 400
    if not datos["tareas_entregadas"].between(0, TOTAL_ACTIVIDADES).all() or not datos[
        "tareas_entregadas"
    ].mod(1).eq(0).all():
        return jsonify(error=f"Las tareas entregadas deben ser un entero entre 0 y {TOTAL_ACTIVIDADES}."), 400
    if not datos["promedio_previo"].between(NOTA_MIN, NOTA_MAX).all():
        return jsonify(error=f"El promedio debe estar entre {NOTA_MIN} y {NOTA_MAX}."), 400
    if not datos[TARGET].isin([0, 1]).all():
        return jsonify(error=f"La columna {TARGET} debe contener únicamente 0 (reprueba) o 1 (aprueba)."), 400

    entradas = datos[INPUT_FEATURES].copy()
    entradas.insert(0, "horas_estudio", HORAS_ESTUDIO)
    reales = datos[TARGET].astype(int)
    filas = []
    metricas = {}
    for nombre, modelo in MODELOS.items():
        predicciones = modelo.predict(entradas).astype(int)
        probabilidades = modelo.predict_proba(entradas)[:, 1]
        matriz = confusion_matrix(reales, predicciones, labels=[0, 1])
        metricas[nombre] = {
            "exactitud": round(float(accuracy_score(reales, predicciones)) * 100, 1),
            "precision": round(float(precision_score(reales, predicciones, zero_division=0)) * 100, 1),
            "sensibilidad": round(float(recall_score(reales, predicciones, zero_division=0)) * 100, 1),
            "f1": round(float(f1_score(reales, predicciones, zero_division=0)) * 100, 1),
            "matriz": matriz.tolist(),
        }
        for i, prediccion in enumerate(predicciones):
            if len(filas) < i + 1:
                filas.append({
                    "asistencia": float(datos.iloc[i]["asistencia"]),
                    "tareas_entregadas": int(datos.iloc[i]["tareas_entregadas"]),
                    "promedio_previo": float(datos.iloc[i]["promedio_previo"]),
                    "real": int(reales.iloc[i]),
                    "predicciones": {},
                })
            filas[i]["predicciones"][nombre] = {
                "aprobado": int(prediccion),
                "probabilidad": round(float(probabilidades[i]), 4),
            }
    return jsonify(total=len(filas), metricas=metricas, filas=filas)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
