# Política de Seguridad

## Reportar Vulnerabilidades

**IMPORTANTE**: No abras issues públicos para reportar vulnerabilidades de seguridad.

Si descubres una vulnerabilidad de seguridad en MeigaSearch, por favor **reporta directamente por email** a:

📧 **meigasearch@example.com**

### Qué incluir en tu reporte

1. **Descripción**: Qué es la vulnerabilidad y cómo impacta
2. **Ubicación**: Archivo, línea de código, componente afectado
3. **Severidad**: Crítica, Alta, Media, Baja
4. **Pasos para reproducir**: Cómo verificar la vulnerabilidad
5. **Propuesta de fix** (opcional): Si tienes una solución

### Ejemplo de reporte

```
Subject: [SECURITY] Vulnerabilidad XSS en búsqueda

Descripción:
El parámetro 'q' en la búsqueda no está sanitizado, permitiendo inyección de JavaScript.

Ubicación:
frontend/index.html, línea ~2350, función performSearch()

Severidad: Alta

Pasos para reproducir:
1. Buscar con query: <img src=x onerror=alert('XSS')>
2. Ver la UI renderizada sin escape

Solución propuesta:
Usar esc() antes de insertar en el DOM o DOMPurify
```

## Proceso de Respuesta

1. **Confirmación**: Recibirás una respuesta en 48 horas
2. **Investigación**: Evaluaremos el reporte y la severidad
3. **Fix**: Trabajaremos en un parche
4. **Coordinación**: Te notificaremos cuando se publique el fix
5. **Disclosure**: Reconocimiento en un advisory de seguridad (si lo deseas)

## Nuestro Compromiso

- ✅ Investigamos todas las vulnerabilidades reportadas
- ✅ No publicaremos detalles hasta que haya un fix
- ✅ Trabajamos rápidamente en severidad crítica
- ✅ Reconocemos a los investigadores responsables

## Prácticas de Seguridad en MeigaSearch

### Autenticación
- JWT tokens con expiración de 24h
- Validación en cada endpoint
- RBAC por departamento

### Datos
- Procesamiento local (sin APIs externas)
- Qdrant con validación de payloads
- Sanitización de inputs antes de buscar

### Dependencias
- Pinned versions en requirements.txt
- Actualizaciones regulares de librerías
- Auditoría con `pip-audit`

## Vulnerabilidades Conocidas

N/A - Proyecto en Hackathon 2026

## Versiones Soportadas

| Versión | Soportada |
| ------- | --------- |
| 1.0.x   | ✅ Sí     |

## Contacto de Seguridad

**Email**: meigasearch@example.com
**GPG**: No disponible en Hackathon
**Respuesta esperada**: 48 horas
