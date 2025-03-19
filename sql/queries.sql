-- CONSULTAS GENERALES

-- 1. Clientes con sus cuentas bancarias
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  cb.cuenta_id,
  cb.tipo_cuenta,
  cb.saldo
FROM clientes c
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id;

-- 2. Transacciones de una cuenta junto con la información del cliente
SELECT 
  c.nombre,
  c.apellido,
  t.transaccion_id,
  t.tipo_transaccion,
  t.monto,
  t.fecha_transaccion
FROM transacciones t
JOIN cuentas_bancarias cb ON t.cuenta_origen_id = cb.cuenta_id
JOIN clientes c ON cb.cliente_id = c.cliente_id;

-- 3. Detalles de préstamos junto con la información del cliente
SELECT 
  p.prestamo_id,
  c.nombre,
  c.apellido,
  p.monto_prestamo,
  p.tasa_interes,
  p.fecha_otorgamiento
FROM prestamos p
JOIN clientes c ON p.cliente_id = c.cliente_id;

-- 4. Pagos de préstamos con información del préstamo y del cliente
SELECT 
  pp.pago_id,
  pp.fecha_pago,
  pp.monto_pagado,
  p.prestamo_id,
  c.nombre,
  c.apellido
FROM pagos_prestamo pp
JOIN prestamos p ON pp.prestamo_id = p.prestamo_id
JOIN clientes c ON p.cliente_id = c.cliente_id;

-- 5. Tarjetas asociadas a cuentas y clientes
SELECT 
  t.tarjeta_id,
  t.tipo_tarjeta,
  t.numero_tarjeta,
  cb.cuenta_id,
  c.nombre,
  c.apellido
FROM tarjetas t
JOIN cuentas_bancarias cb ON t.cuenta_id = cb.cuenta_id
JOIN clientes c ON cb.cliente_id = c.cliente_id;

-- 6. Cantidad de cuentas por cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  COUNT(cb.cuenta_id) AS num_cuentas
FROM clientes c
LEFT JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- 7. Suma de montos de transacciones por cuenta
SELECT 
  cb.cuenta_id,
  SUM(t.monto) AS total_transacciones
FROM cuentas_bancarias cb
JOIN transacciones t ON cb.cuenta_id = t.cuenta_origen_id
GROUP BY cb.cuenta_id;

-- 8. Total de préstamos por cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  SUM(p.monto_prestamo) AS total_prestamos
FROM clientes c
LEFT JOIN prestamos p ON c.cliente_id = p.cliente_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- 9. Clientes con préstamo pero sin tarjeta de crédito
SELECT DISTINCT 
  c.cliente_id,
  c.nombre,
  c.apellido
FROM clientes c
JOIN prestamos p ON c.cliente_id = p.cliente_id
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
LEFT JOIN tarjetas t ON cb.cuenta_id = t.cuenta_id AND t.tipo_tarjeta = 'crédito'
WHERE t.tarjeta_id IS NULL;

-- 10. Última transacción de cada cuenta de cada cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  cb.cuenta_id,
  MAX(t.fecha_transaccion) AS ultima_transaccion
FROM clientes c
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
JOIN transacciones t ON cb.cuenta_id = t.cuenta_origen_id
GROUP BY c.cliente_id, c.nombre, c.apellido, cb.cuenta_id;

-- 11. Cuentas sin transacciones registradas
SELECT 
  cb.cuenta_id,
  c.nombre,
  c.apellido
FROM cuentas_bancarias cb
JOIN clientes c ON cb.cliente_id = c.cliente_id
LEFT JOIN transacciones t ON cb.cuenta_id = t.cuenta_origen_id
WHERE t.transaccion_id IS NULL;

-- 12. Total mensual de pagos de préstamos
SELECT 
  date_trunc('month', pp.fecha_pago) AS mes,
  SUM(pp.monto_pagado) AS total_pago
FROM pagos_prestamo pp
GROUP BY mes
ORDER BY mes;

-- 13. Transacciones de tipo 'retiro' con datos del cliente
SELECT 
  t.transaccion_id,
  t.monto,
  t.fecha_transaccion,
  c.nombre,
  c.apellido
FROM transacciones t
JOIN cuentas_bancarias cb ON t.cuenta_origen_id = cb.cuenta_id
JOIN clientes c ON cb.cliente_id = c.cliente_id
WHERE t.tipo_transaccion = 'retiro';

-- 14. Clientes con más de una cuenta
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  COUNT(cb.cuenta_id) AS total_cuentas
FROM clientes c
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
GROUP BY c.cliente_id, c.nombre, c.apellido
HAVING COUNT(cb.cuenta_id) > 1;

-- 15. Detalle de pagos por préstamo (pagos completados y pendientes)
SELECT 
  p.prestamo_id,
  c.nombre,
  c.apellido,
  SUM(CASE WHEN pp.estado_pago = 'completado' THEN 1 ELSE 0 END) AS pagos_completados,
  SUM(CASE WHEN pp.estado_pago <> 'completado' THEN 1 ELSE 0 END) AS pagos_pendientes
FROM prestamos p
JOIN clientes c ON p.cliente_id = c.cliente_id
LEFT JOIN pagos_prestamo pp ON p.prestamo_id = pp.prestamo_id
GROUP BY p.prestamo_id, c.nombre, c.apellido;

-- 16. Detalles de tarjetas de crédito y saldo de la cuenta asociada
SELECT 
  t.tarjeta_id,
  t.numero_tarjeta,
  t.tipo_tarjeta,
  cb.saldo,
  c.nombre,
  c.apellido
FROM tarjetas t
JOIN cuentas_bancarias cb ON t.cuenta_id = cb.cuenta_id
JOIN clientes c ON cb.cliente_id = c.cliente_id
WHERE t.tipo_tarjeta = 'crédito';

-- 17. Total de transacciones realizadas por cada cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  SUM(t.monto) AS total_transacciones
FROM clientes c
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
JOIN transacciones t ON cb.cuenta_id = t.cuenta_origen_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- 18. Cuentas y sus tarjetas asociadas (si existen)
SELECT 
  cb.cuenta_id,
  c.nombre,
  c.apellido,
  t.tarjeta_id,
  t.tipo_tarjeta,
  t.numero_tarjeta
FROM cuentas_bancarias cb
JOIN clientes c ON cb.cliente_id = c.cliente_id
LEFT JOIN tarjetas t ON cb.cuenta_id = t.cuenta_id
ORDER BY cb.cuenta_id;

-- 19. Transacciones con descripción y correo electrónico del cliente
SELECT 
  t.transaccion_id,
  t.descripcion,
  c.email,
  t.fecha_transaccion
FROM transacciones t
JOIN cuentas_bancarias cb ON t.cuenta_origen_id = cb.cuenta_id
JOIN clientes c ON cb.cliente_id = c.cliente_id;

-- 20. Total de pagos realizados en préstamos por cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  SUM(pp.monto_pagado) AS total_pagado
FROM clientes c
JOIN prestamos p ON c.cliente_id = p.cliente_id
JOIN pagos_prestamo pp ON p.prestamo_id = pp.prestamo_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- CONSULTAS BUSINESS INTELLIGENCE

-- Evolución Mensual del Volumen y Cantidad de Transacciones
SELECT
    DATE_TRUNC('month', fecha_transaccion) AS mes,
    COUNT(*) AS total_transacciones,
    SUM(monto) AS volumen_total
FROM transacciones
GROUP BY DATE_TRUNC('month', fecha_transaccion)
ORDER BY mes;

-- Segmentación de Clientes Según Actividad Bancaria
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  COUNT(t.transaccion_id) AS total_transacciones,
  CASE 
    WHEN COUNT(t.transaccion_id) < 5 THEN 'Baja'
    WHEN COUNT(t.transaccion_id) BETWEEN 5 AND 15 THEN 'Media'
    ELSE 'Alta'
  END AS segmento_actividad
FROM clientes c
JOIN cuentas_bancarias cb ON c.cliente_id = cb.cliente_id
JOIN transacciones t ON cb.cuenta_id = t.cuenta_origen_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- Análisis de Préstamos: Monto Otorgado por Mes
SELECT 
  DATE_TRUNC('month', fecha_otorgamiento) AS mes,
  COUNT(*) AS total_prestamos,
  SUM(monto_prestamo) AS total_otorgado,
  AVG(monto_prestamo) AS promedio_prestamo,
  MAX(monto_prestamo) AS max_prestamo,
  MIN(monto_prestamo) AS min_prestamo
FROM prestamos
GROUP BY DATE_TRUNC('month', fecha_otorgamiento)
ORDER BY mes;

-- Ratio de Cumplimiento de Pagos en Préstamos por Cliente
SELECT 
  c.cliente_id,
  c.nombre,
  c.apellido,
  COUNT(pp.pago_id) FILTER (WHERE pp.estado_pago = 'completado') AS pagos_completados,
  COUNT(pp.pago_id) AS total_pagos,
  ROUND(
    (COUNT(pp.pago_id) FILTER (WHERE pp.estado_pago = 'completado')::decimal / NULLIF(COUNT(pp.pago_id), 0)) * 100, 2
  ) AS porcentaje_cumplimiento
FROM clientes c
JOIN prestamos p ON c.cliente_id = p.cliente_id
JOIN pagos_prestamo pp ON p.prestamo_id = pp.prestamo_id
GROUP BY c.cliente_id, c.nombre, c.apellido;

-- Distribución de Saldos de Cuentas Según la Presencia de Tarjeta de Crédito
SELECT 
  CASE 
    WHEN t.tarjeta_id IS NULL THEN 'Sin tarjeta de crédito'
    ELSE 'Con tarjeta de crédito'
  END AS estado_tarjeta_credito,
  COUNT(DISTINCT cb.cuenta_id) AS total_cuentas,
  AVG(cb.saldo) AS saldo_promedio
FROM cuentas_bancarias cb
LEFT JOIN (
    SELECT DISTINCT cuenta_id, tarjeta_id
    FROM tarjetas
    WHERE tipo_tarjeta = 'crédito'
) t ON cb.cuenta_id = t.cuenta_id
GROUP BY estado_tarjeta_credito;

