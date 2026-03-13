#!/usr/bin/env bash
# Equivalente Linux de run_ram_compare.ps1
# Perfila uso de RAM con mprof (memory_profiler) para cada combinación
# de dimensión, stream_step y método.
#
# Uso:
#   bash run_ram_compare.sh [opciones]
#
# Opciones (variables de entorno o editar defaults aquí abajo):
#   STREAM_STEPS  Lista de steps separados por espacio  (default: "4")
#   DIMENSIONS    Lista de dimensiones                  (default: "2")
#   METHODS       Lista de métodos                      (default: "uniform kdtree")
#   PYTHON_SCRIPT Ruta al script Python                 (default: ./main_debug.py)
#   OUTPUT_DIR    Directorio de salida                  (default: ./ram_profiles)
#
# Ejemplo con overrides:
#   DIMENSIONS="2 4" STREAM_STEPS="4 8" bash run_ram_compare.sh

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
STREAM_STEPS="${STREAM_STEPS:-4}"
DIMENSIONS="${DIMENSIONS:-2}"
METHODS="${METHODS:-uniform kdtree}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-./main_debug.py}"
OUTPUT_DIR="${OUTPUT_DIR:-./ram_profiles}"

# ── Validaciones ────────────────────────────────────────────────────────────
if ! command -v mprof &>/dev/null; then
    echo "ERROR: mprof no encontrado. Instala memory_profiler: pip install memory_profiler"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Script Python no encontrado: $PYTHON_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SUMMARY_FILE="$OUTPUT_DIR/ram_summary.csv"
echo "dim,stream_step,method,peak_mib,profile_file,peak_file" > "$SUMMARY_FILE"

echo "Running RAM profiling per configuration"
echo "  Dimensions  : $DIMENSIONS"
echo "  StreamSteps : $STREAM_STEPS"
echo "  Methods     : $METHODS"
echo "  OutputDir   : $OUTPUT_DIR"

# ── Bucle principal ─────────────────────────────────────────────────────────
for dim in $DIMENSIONS; do
    for step in $STREAM_STEPS; do
        for method in $METHODS; do

            CONFIG_DIR="$OUTPUT_DIR/dim_${dim}/step_${step}/method_${method}"
            mkdir -p "$CONFIG_DIR"

            DAT_FILE="$CONFIG_DIR/profile.dat"
            PEAK_FILE="$CONFIG_DIR/peak.txt"

            echo ""
            echo "Running dim=$dim, step=$step, method=$method"

            mprof run --include-children --output "$DAT_FILE" \
                python "$PYTHON_SCRIPT" \
                "dim=$dim" \
                "stream_steps=[$step]" \
                "methods_to_test=[$method]"

            # Capturar pico de RAM
            PEAK_OUTPUT=$(mprof peak "$DAT_FILE" 2>&1 || true)
            echo "$PEAK_OUTPUT" > "$PEAK_FILE"

            PEAK_MIB=""
            if [[ "$PEAK_OUTPUT" =~ ([0-9]+(\.[0-9]+)?)[[:space:]]*MiB ]]; then
                PEAK_MIB="${BASH_REMATCH[1]}"
            fi

            echo "$dim,$step,$method,$PEAK_MIB,$DAT_FILE,$PEAK_FILE" >> "$SUMMARY_FILE"

            echo "Peak RAM : ${PEAK_MIB:-N/A} MiB"
            echo "Saved in : $CONFIG_DIR"
        done
    done
done

# ── Resumen final ────────────────────────────────────────────────────────────
echo ""
echo "=== CONFIG SUMMARY (MiB) ==="
column -t -s',' "$SUMMARY_FILE"
echo ""
echo "Saved: $SUMMARY_FILE"
