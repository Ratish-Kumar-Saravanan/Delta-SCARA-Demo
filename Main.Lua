RobotServoOn()

-- Tune these
local jog_xy_speed = 1000  
local jog_z_speed  = 2000  
local move_speedL  = 450 
local move_speedP  = 60    
local xp,yp;



local x_min, x_max = 380, 588
local y_min, y_max = -140, 340


local nest_x, nest_y = 550, 120
local home_x, home_y = 250, 125
local z_safe, z_pick, z_nest = 0, -180, -120


local x, y, z = 300, 80, -20
local x_home, y_home, z_home = x, y, z


local jogging_axis = ""  

local function start_jog(dir, spd)
  if jogging_axis ~= dir then
    MotionStop()                    
    ContinueCartesianJOG(dir, spd)  
    jogging_axis = dir
  end
end

local function stop_jog()
  if jogging_axis ~= "" then
    MotionStop()
    jogging_axis = ""
  end
end

while 1 do

  local x_plus   = ReadModbus(0x1000, "W")
  local x_minus  = ReadModbus(0x1001, "W")
  local y_plus   = ReadModbus(0x1002, "W")
  local y_minus  = ReadModbus(0x1003, "W")
  local z_drop   = ReadModbus(0x1004, "W")
  local home_loc = ReadModbus(0x1005, "W")


  if     x_plus  == 1 and x < x_max  then start_jog("X+", jog_xy_speed)
  elseif x_minus == 1 and x > x_min  then start_jog("X-", jog_xy_speed)
  elseif y_plus  == 1 and y < y_max  then start_jog("Y+", jog_xy_speed)
  elseif y_minus == 1 and y > y_min  then start_jog("Y-", jog_xy_speed)
  else
    stop_jog()
  end


  x=RobotX()
  y=RobotY()
  if x > x_max or x < x_min or y > y_max or y < y_min then
    WriteModbus(0x1007, "W", 1)
    DELAY(0.1)
    WriteModbus(0x1007, "W", 0)
  end

  -- -------- Home --------
  if home_loc == 1 then
    stop_jog()
    MovP(X(x_home) + Y(y_home) + Z(z_home), SPD(move_speedP))
  end

  -- -------- Pick / place sequence --------
  if z_drop == 1 then
    stop_jog()

    -- Up to safe
    xp = RobotX()
    yp = RobotY()

    MovL(X(xp) + Y(yp) + Z(z_safe), SPD(move_speedL))

    -- Down to pick
    MovL(X(xp) + Y(yp) + Z(z_pick), SPD(move_speedL))

    DO(1, ON)  -- vacuum on

    -- Back up
    MovL(X(xp) + Y(yp) + Z(z_safe), SPD(move_speedL))

    -- To nest
    MovL(X(nest_x) + Y(nest_y) + Z(z_safe), SPD(move_speedL))
    MovL(X(nest_x) + Y(nest_y) + Z(z_nest), SPD(move_speedL))

    DO(2, ON) ; DELAY(0.2) ; DO(1, OFF) ; DO(2, OFF)

    MovL(X(nest_x) + Y(nest_y) + Z(z_safe), SPD(move_speedL))
    MovL(X(home_x) + Y(home_y) + Z(z_safe), SPD(move_speedL))
  end
end

RobotServoOff()
