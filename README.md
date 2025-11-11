# Visualizador de Series de Fourier

## Introducción

Aplicación interactiva en Python para visualizar series de Fourier y el fenómeno de beats. Permite explorar cómo diferentes señales periódicas se aproximan mediante sumas de funciones senoidales y cosenoidales, con visualizaciones en tiempo real, cálculo de error cuadrático medio (MSE) y representación de fórmulas matemáticas.

## Objetivos de Aprendizaje

- Comprender visualmente cómo las series de Fourier aproximan señales periódicas
- Analizar la convergencia mediante el error cuadrático medio (MSE)
- Comparar diferentes tipos de señales y sus tasas de convergencia
- Explorar el fenómeno de beats e interferencia entre ondas

### System Requirements

- **Sistema Operativo**: Windows, macOS o Linux
- **Python**: 3.7 o superior
- **RAM**: Mínimo 2 GB
- **Espacio en disco**: < 50 MB

### Instalación

0. En una carpeta clone el proyecto de GitHub.

1. Verifica que tengas Python 3.7+ instalado:
   ```bash
   python --version
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
   O manualmente:
   ```bash
   pip install numpy matplotlib sympy
   ```

3. Ejecuta la aplicación:
   ```bash
   python main.py
   ```

### Overview

La interfaz consta de:
- **Menú desplegable**: Selecciona el tipo de señal
- **Controles**: Slider para N (Fourier) o controles de frecuencia/amplitud (Beats)
- **Gráficas**: Visualización de la aproximación y señal objetivo (Fourier) o las tres señales (Beats)
- **Fórmulas**: Representación matemática en notación compacta y expandida

### Señales

**Onda Cuadrada**: Armónicos impares, coeficientes ∝ 1/n. Convergencia lenta debido a discontinuidades.

**Onda Triangular**: Armónicos impares, coeficientes ∝ 1/n². Convergencia más rápida que la cuadrada.

**Diente de Sierra**: Todos los armónicos, coeficientes ∝ (-1)^n/n. Discontinuidad en cada periodo.

**Parábola**: Todos los armónicos en coseno, término constante. Coeficientes ∝ (-1)^n/n². Buena convergencia.

**Beats**: Visualización de interferencia entre dos ondas senoidales con frecuencias diferentes, mostrando modulación de amplitud.

### Ejemplo de Uso

1. Selecciona "Cuadrada" en el menú
2. Ajusta N comenzando con valor bajo (N=3)
3. Incrementa N gradualmente (5, 7, 10, 15) y observa la mejora en la aproximación
4. Observa cómo el MSE disminuye al aumentar N
5. Compara con otras señales para notar diferencias en la convergencia

Para Beats:
1. Selecciona "Beats"
2. Ajusta las frecuencias f₁ y f₂
3. Observa cómo la diferencia de frecuencias afecta el periodo de beats
4. Modifica las amplitudes para ver su efecto en la envolvente

### Recomendaciones de Uso

- Comienza con N bajo (3-5) para entender la construcción término por término
- Observa el MSE como métrica cuantitativa de la calidad de la aproximación
- Compara visualmente la aproximación con la señal objetivo
- Experimenta con diferentes valores de frecuencia en Beats para comprender la relación f_beat = |f₂ - f₁|
- Valores de N entre 5-20 son suficientes para la mayoría de propósitos educativos.
- Para más detalle sobre el fundamento matematico chechar el pdf del mismo nombre.
- Lea el manual de usuario antes de usar.