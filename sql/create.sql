CREATE TABLE clientes (
  cliente_id SERIAL PRIMARY KEY,
  nombre VARCHAR(50) NOT NULL,
  apellido VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE,
  direccion VARCHAR(255),
  telefono VARCHAR(20),
  fecha_registro DATE NOT NULL
);


CREATE TABLE cuentas_bancarias (
  cuenta_id SERIAL PRIMARY KEY,
  cliente_id INT NOT NULL,
  tipo_cuenta VARCHAR(20) NOT NULL, -- Ej: 'ahorros', 'corriente'
  saldo DECIMAL(15,2) DEFAULT 0.00,
  fecha_apertura DATE NOT NULL,
  estado VARCHAR(20) DEFAULT 'activa',
  FOREIGN KEY (cliente_id) REFERENCES clientes(cliente_id)
);



CREATE TABLE transacciones (
  transaccion_id SERIAL PRIMARY KEY,
  cuenta_origen_id INT,
  cuenta_destino_id INT,
  monto DECIMAL(15,2) NOT NULL,
  fecha_transaccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  tipo_transaccion VARCHAR(20) NOT NULL, -- Ej: 'depósito', 'retiro', 'transferencia'
  descripcion VARCHAR(255),
  FOREIGN KEY (cuenta_origen_id) REFERENCES cuentas_bancarias(cuenta_id),
  FOREIGN KEY (cuenta_destino_id) REFERENCES cuentas_bancarias(cuenta_id)
);



CREATE TABLE prestamos (
  prestamo_id SERIAL PRIMARY KEY,
  cliente_id INT NOT NULL,
  monto_prestamo DECIMAL(15,2) NOT NULL,
  tasa_interes DECIMAL(5,2) NOT NULL,
  plazo INT NOT NULL, -- en meses
  fecha_otorgamiento DATE NOT NULL,
  saldo_restante DECIMAL(15,2) NOT NULL,
  FOREIGN KEY (cliente_id) REFERENCES clientes(cliente_id)
);


CREATE TABLE pagos_prestamo (
  pago_id SERIAL PRIMARY KEY,
  prestamo_id INT NOT NULL,
  fecha_pago DATE NOT NULL,
  monto_pagado DECIMAL(15,2) NOT NULL,
  estado_pago VARCHAR(20) DEFAULT 'pendiente', -- Ej: 'completado', 'pendiente', 'fallido'
  FOREIGN KEY (prestamo_id) REFERENCES prestamos(prestamo_id)
);



CREATE TABLE tarjetas (
  tarjeta_id SERIAL PRIMARY KEY,
  cuenta_id INT NOT NULL,
  tipo_tarjeta VARCHAR(20) NOT NULL, -- Ej: 'débito', 'crédito'
  numero_tarjeta VARCHAR(20) NOT NULL,  -- En entornos reales se debe almacenar de forma segura (tokenización o cifrado)
  fecha_emision DATE NOT NULL,
  fecha_expiracion DATE NOT NULL,
  estado_tarjeta VARCHAR(20) DEFAULT 'activa',
  FOREIGN KEY (cuenta_id) REFERENCES cuentas_bancarias(cuenta_id)
);

commit;


