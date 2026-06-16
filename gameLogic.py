import random
import math
nonEnemyObjects = ["Stimpack", "Medikit", "HealthBonus", "ClipBox", "DoomPlayer", "BlueArmor", "ShellBox", "TeleportFog", "BaronBall", "BulletPuff", "Blood"]
distanceUntilTooClose = 200
ticksFor5Seconds = 175
lastTickHurtCalled = -200
previousHealth = 100
tickOfEnd = 4200

def lowHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health <= 33.3

def mediumHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health > 33 and health <= 66.6

def highHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health > 66.6

def lowArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor <= 33.3

def mediumArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor > 33 and armor <= 66.6

def highArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor > 66.6

def lowAmmoCurrent(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # if super shotgun
    if(currentWeapon == 3):
        return firstWepAmmo < 10
    # otherwise, chaingun
    return secondWepAmmo < 40

def wieldingChaingun(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return currentWeapon == 4

def nearbyEnemy(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def recentlyHurt(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    if(health < previousHealth):
        previousHealth = health
        if(currentTick - lastTickHurtCalled < ticksFor5Seconds):
            lastTickHurtCalled = currentTick
            return True
    lastTickHurtCalled = currentTick
    return False

def healthNearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "Medikit"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo3Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ShellBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo4Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ClipBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def armorNearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "BlueArmor"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def manyEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    count = 0
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           count += 1
    
    return count >= 6

def noEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    count = 0
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           count += 1
    
    return count == 0

def someRangedEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    numRanged = 0
    for object in state.objects:
       if(object.name == "ChaingunGuy" or object.name == "HellKnight"):
           numRanged += 1
    
    return numRanged >= 2

def lowTimeRemaining(state, currentTick):
    return currentTick >= (tickOfEnd - 500)

def getClosestEnemyPosition(currentX, currentY, state):
    closestDistance = None
    closestDistanceCoords = (0, 0)
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           playerDistance = ((object.position_x - currentX)**2 + (object.posiition_y - currentY) **2) ** (1/2)
           if(closestDistance == None or playerDistance < closestDistance):
               closestDistance = playerDistance
               closestDistanceCoords = (object.position_x, object.position_y)

    if(closestDistance == None):
        return False, None, None
    return True, closestDistanceCoords[0], closestDistanceCoords[1]

actionMapping = {
    "attack": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "move_right": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "move_left": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "move_backward": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "move_forward": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "turn_right": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "turn_left": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "select_shotgun": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "select_chaingun": [0, 0, 0, 0, 0, 0, 0, 0, 1]
}

previousAction = [0, 0, 0, 0, 0, 0, 0, 0, 0]

class RealActions:
    def __init__(self):
        self.finished = True
    def activateAction(self):
        self.finished = False
    def deactivateAction(self):
        self.finished = True
    def findDirectionToFaceObject(self, playerObjX, playerObjY, playerAngle, targetObjX, targetObjY):
        offsetVector = (targetObjX - playerObjX, targetObjY - playerObjY)

        objectDeg = math.degrees(math.atan2(offsetVector[1], offsetVector[0])) + 180

        angleClockwise = abs(objectDeg - playerAngle)

        if(angleClockwise > 180):
            angleMagnitude = 360 - angleClockwise
            if(objectDeg - playerAngle > 0):
                return (angleMagnitude, actionMapping["turn_right"])
            return (angleMagnitude, actionMapping["turn_left"])
        else:
            if(objectDeg - playerAngle > 0):
                return (angleClockwise, actionMapping["turn_left"])
            return (angleClockwise, actionMapping["turn_right"])
        
    '''
    meaning of targetVector parameter:
    0 = wants to move towards them
    90 = wants to move to left of them
    180 = want to move away from em
    270 = want to move to right of them 
    ...and everything in between for more nuanced movement
    '''
    def findMovementToMoveRelativeToObject(self, playerObjX, playerObjY, playerAngle, targetObjX, targetObjY, angleOffset):
        (trueAngleVal, trueActionVector) = self.findDirectionToFaceObject(playerObjX, playerObjY, playerAngle, targetObjX, targetObjY)
        (tempAngleVal, tempAngleDirection) = self.findDirectionToFaceObject(playerObjX, playerObjY, (playerAngle + angleOffset) % 360, targetObjX, targetObjY)

        
        leftRightMovementFactor = actionMapping["move_left"] if tempAngleDirection == actionMapping["turn_left"] else actionMapping["move_right"]
        if(tempAngleVal < 1):
            trueActionVector += actionMapping["move_forward"]
        elif(tempAngleVal < 89):
            trueActionVector += leftRightMovementFactor + actionMapping["move_forward"]
        elif(tempAngleVal < 91):
            trueActionVector += leftRightMovementFactor
        elif(tempAngleVal < 179):
            trueActionVector += leftRightMovementFactor + actionMapping["move_backward"]
        else:
            trueActionVector += actionMapping["move_backward"]

        return trueActionVector, trueAngleVal

class SwitchWeapon(RealActions):
    def activateAction(self):
        self.calledWeaponSwitch = False
        super().activateAction()
        
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        if not self.calledWeaponSwitch:
            self.weaponToSwitchTo = "chaingun" if currentWeapon == 3 else "shotgun"
            self.calledWeaponSwitch = True
            return previousAction + actionMapping[f"select_{self.weaponToSwitchTo}"]
        
        if((self.weaponToSwitchTo == "chaingun" and self.currentWeapon == 4) or (self.weaponToSwitchTo == "shotgun" and self.currentWeapon == 3)):
            self.deactivateAction()

        return previousAction
                
TICKS_FIREANDSTRAFE_IS_ACTIVE = 100
class FireAndStrafe(RealActions):
    def activateAction(self):
        self.strafeDirection = 90 if random.random() > 0.5 else 270
        self.currentTick = 0
        super().activateAction()
    def updateTickAndReturnAction(self, state):        
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        success, enemyX, enemyY = getClosestEnemyPosition(posX, posY, state)
        if(not success):
            self.deactivateAction()
            return previousAction
        
        chosenAction, angle = self.findMovementToMoveRelativeToObject(posX, posY, angle, enemyX, enemyY, self.strafeDirection)

        
        if(angle < 1):
            chosenAction += actionMapping["attack"]

        self.currentTick += 1
        if(self.currentTick >= TICKS_FIREANDSTRAFE_IS_ACTIVE):
            self.deactivateAction()

        previousAction = chosenAction
        return chosenAction

class DirectlyFlee(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class GoToHealth(RealActions):
    def updateTickAndReturnAction(self,state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class GoToAmmo(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class GoToArmor(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class MoveRandom(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class RunAway(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        pass

class ChargeIn(RealActions):
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
         
        pass