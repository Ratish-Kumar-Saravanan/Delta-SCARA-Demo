RobotServoOn()

--setting jog speed,acc and dec
SpdJ(150)       
AccJ(150)       
DecJ(150)       

--setting linear acc,speed and dec
SpdL(150)      
AccL(150)     
DecL(150)     

--setting local limits,nest and home positions
local x_min, x_max = 380, 588
local y_min, y_max = -140, 340

local nest_x, nest_y = 550, 120
local home_x, home_y = 250, 125

local z_safe, z_pick, z_nest = 0, -180, -120

-- local home reference
local x, y, z = 300, 80, -20
local x_home, y_home, z_home = x, y, z

local jogging_axis = "" 

local function start_jog(dir)
  if jogging_axis ~= dir then
    MotionStop()
    ContinueCartesianJOG(dir, 0)
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

  local x_plus      = ReadModbus(0x1000, "W")
  local x_minus     = ReadModbus(0x1001, "W")
  local y_plus      = ReadModbus(0x1002, "W")
  local y_minus     = ReadModbus(0x1003, "W")
  local z_drop      = ReadModbus(0x1004, "W")
  local home_loc    = ReadModbus(0x1005, "W")
  local coordinates = ReadModbus(0x100A, "W")
  local X_pos       = ReadModbus(0x1008, "W")
  local Y_pos       = ReadModbus(0x1009, "W")

  -- -------- Strat jogging if within limits with set SpdJ --------
  if     x_plus  == 1 and x < x_max  then start_jog("X+")
  elseif x_minus == 1 and x > x_min  then start_jog("X-")
  elseif y_plus  == 1 and y < y_max  then start_jog("Y+")
  elseif y_minus == 1 and y > y_min  then start_jog("Y-")
  else
    stop_jog()
  end


  -- reading x and y for current robot position
  x = RobotX()
  y = RobotY()
  if x > x_max or x < x_min or y > y_max or y < y_min then
    WriteModbus(0x1007, "W", 1)   
    DELAY(0.1)
    WriteModbus(0x1007, "W", 0)
  end

 
  if coordinates == 1 then

    MovL(X(X_pos) + Y(Y_pos) + Z(z_home))
  end


  if home_loc == 1 then
    stop_jog()
    MovL(X(x_home) + Y(y_home) + Z(z_home))
  end

  
  if z_drop == 1 then
    stop_jog()

-- if the pickup button is pressed move to xp,yp,zsafe and then pick up ball by going to zpick and then come back and move to nest position and drop the ball
    local xp = RobotX()
    local yp = RobotY()

    MovL(X(xp) + Y(yp) + Z(z_safe))

  
    MovL(X(xp) + Y(yp) + Z(z_pick))
    DO(1, ON)            


    MovL(X(xp) + Y(yp) + Z(z_safe))

    
    MovL(X(nest_x) + Y(nest_y) + Z(z_safe))
    MovL(X(nest_x) + Y(nest_y) + Z(z_nest))

 
    DO(2, ON) ; DELAY(0.2) ; DO(1, OFF) ; DO(2, OFF)


    MovL(X(nest_x) + Y(nest_y) + Z(z_safe))
    MovL(X(home_x) + Y(home_y) + Z(z_safe))
  end

  DELAY(0.02)
end

RobotServoOff()
