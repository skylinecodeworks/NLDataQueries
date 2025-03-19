DO $$
DECLARE
  i INT;
  numTrans INT;
  j INT;
  randomMonto NUMERIC;
  clienteId INT;
  cuentaId INT;
  prestamoId INT;
BEGIN
  FOR i IN 1..100 LOOP
    -- Insertar cliente
    INSERT INTO clientes (nombre, apellido, email, direccion, telefono, fecha_registro)
    VALUES (
      'Nombre' || i,
      'Apellido' || i,
      'cliente' || i || '@example.com',
      'Dirección ' || i,
      '900000' || lpad(i::text, 5, '0'),
      current_date
    )
    RETURNING cliente_id INTO clienteId;
    
    -- Insertar cuenta bancaria asociada al cliente
    INSERT INTO cuentas_bancarias (cliente_id, tipo_cuenta, saldo, fecha_apertura, estado)
    VALUES (
      clienteId,
      'ahorros',
      round((random() * 10000)::numeric, 2),
      current_date,
      'activa'
    )
    RETURNING cuenta_id INTO cuentaId;
    
    -- Generar entre 1 y 5 transacciones para la cuenta
    SELECT floor(1 + random() * 5)::int INTO numTrans;
    FOR j IN 1..numTrans LOOP
      randomMonto := round((50 + random() * 500)::numeric, 2);
      INSERT INTO transacciones (cuenta_origen_id, cuenta_destino_id, monto, fecha_transaccion, tipo_transaccion, descripcion)
      VALUES (
        cuentaId,
        cuentaId,
        randomMonto,
        now(),
        'depósito',
        'Transacción ' || j || ' para cuenta ' || cuentaId
      );
    END LOOP;
    
    -- Con probabilidad del 30%, insertar un préstamo y 12 pagos mensuales
    IF random() < 0.3 THEN
      randomMonto := round((1000 + random() * 9000)::numeric, 2);
      INSERT INTO prestamos (cliente_id, monto_prestamo, tasa_interes, plazo, fecha_otorgamiento, saldo_restante)
      VALUES (
        clienteId,
        randomMonto,
        round((5 + random() * 10)::numeric, 2),
        12,
        current_date,
        randomMonto
      )
      RETURNING prestamo_id INTO prestamoId;
      
      FOR j IN 1..12 LOOP
        INSERT INTO pagos_prestamo (prestamo_id, fecha_pago, monto_pagado, estado_pago)
        VALUES (
          prestamoId,
          current_date + (j * interval '1 month'),
          round(randomMonto / 12, 2),
          'completado'
        );
      END LOOP;
    END IF;
    
    -- Insertar tarjeta de débito asociada a la cuenta
    INSERT INTO tarjetas (cuenta_id, tipo_tarjeta, numero_tarjeta, fecha_emision, fecha_expiracion, estado_tarjeta)
    VALUES (
      cuentaId,
      'débito',
      '4000' || lpad(i::text, 12, '0'),
      current_date,
      current_date + interval '3 years',
      'activa'
    );
    
    -- Con probabilidad del 20%, insertar una tarjeta de crédito adicional
    IF random() < 0.2 THEN
      INSERT INTO tarjetas (cuenta_id, tipo_tarjeta, numero_tarjeta, fecha_emision, fecha_expiracion, estado_tarjeta)
      VALUES (
        cuentaId,
        'crédito',
        '5000' || lpad(i::text, 12, '0'),
        current_date,
        current_date + interval '3 years',
        'activa'
      );
    END IF;
    
  END LOOP;
END$$;
