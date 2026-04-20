-- 1. Загрузка модуля MAVLink
local mav = require("MAVLink/mavlink_msgs")

-- 2. Инициализация приёма команд
mavlink:init(5, 5) -- буфер на 5 сообщений
mavlink:register_rx_msgid(mav.get_msgid("COMMAND_LONG")) -- приём команд

-- 3. Основная функция обработки команд
function handle_command(cmd)
    -- Проверяем, что это наша пользовательская команда (ID 31000)
    if cmd.command == 31000 then
        -- Получаем параметры смещения:
        -- param1 = смещение на север (м)
        -- param2 = смещение на восток (м)
        local offset_north = cmd.param1
        local offset_east = cmd.param2
        
        -- Получаем текущую позицию
        local current_pos = ahrs:get_position()
        if not current_pos then
            gcs:send_text(0, "Ошибка: нет данных позиции")
            return
        end
        
        -- Создаём новую целевую позицию со смещением
        local target = current_pos:copy()
        target:change_alt_frame(0)
        target:offset(offset_north, offset_east)
        
        -- Переключаем в GUIDED режим и летим к цели
        vehicle:set_mode(15) -- 15 = GUIDED режим
        vehicle:set_target_location(target)
        
        gcs:send_text(0, string.format("Смещение: %.1f м N, %.1f м E", 
                      offset_north, offset_east))
    end
end

-- 4. Главный цикл
function update()
    -- Проверяем входящие сообщения
    local msg = mavlink:receive_chan()
    if msg then
        local cmd = mav.decode(msg, {[mav.get_msgid("COMMAND_LONG")] = "COMMAND_LONG"})
        if cmd then handle_command(cmd) end
    end
    
    return update, 500 -- Проверяем команды каждые 500мс
end

-- Запускаем скрипт
gcs:send_text(0, "Скрипт смещения активирован")
return update()