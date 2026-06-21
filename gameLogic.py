import random
import math

import numpy as np
nonEnemyObjects = ["Stimpack", "Medikit", "HealthBonus", "ClipBox", "DoomPlayer", "BlueArmor", "ShellBox", "TeleportFog", "BaronBall", "BulletPuff", "Blood"]
distanceUntilTooClose = 200
ticksFor5Seconds = 175
lastTickHurtCalled = -200
previousHealth = 100
tickOfEnd = 4200
xBounds = [-225, 1225]
yBounds = [-225, 1250]

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
           playerDistance = ((object.position_x - posX)**2 + (object.position_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def recentlyHurt(state, currentTick):
    global previousHealth, lastTickHurtCalled, previousAction
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
           playerDistance = ((object.position_x - posX)**2 + (object.position_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo3Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ShellBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.position_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo4Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ClipBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.position_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def armorNearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "BlueArmor"):
           playerDistance = ((object.position_x - posX)**2 + (object.position_y - posY) **2) ** (1/2)
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

def evaluatedCheckedCondition(object, condition):
    if(condition == "NearestEnemy"):
        return object.name not in nonEnemyObjects
    elif(condition == "NearestMedkit"):
        return object.name == "Medikit"
    elif(condition == "shells"):
        return object.name == "ShellBox"
    elif(condition == "clips"):
        return object.name == "ClipBox"
    elif(condition == "armor"):
        return object.name == "BlueArmor"

def getClosestObjPosition(currentX, currentY, state, checkedCondition):
    closestDistanceCoords = (0, 0)
    closestDistance = None
    objectCountPassing = 0
    for object in state.objects:
       if(evaluatedCheckedCondition(object, checkedCondition)):
           objectCountPassing += 1
           playerDistance = ((object.position_x - currentX)**2 + (object.position_y - currentY) **2) ** (1/2)
           if(closestDistance == None or playerDistance < closestDistance):
               closestDistance = playerDistance
               closestDistanceCoords = (object.position_x, object.position_y)

    return objectCountPassing, closestDistanceCoords[0], closestDistanceCoords[1]

actionMapping = {
    "attack": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "move_right": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "move_left": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "move_backward": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "move_forward": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "turn_right": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "turn_left": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "select_shotgun": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "select_chaingun": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
}

previousAction = [0, 0, 0, 0, 0, 0, 0, 0, 0]

class RealActions:
    def __init__(self):
        self.finished = True
    def activateAction(self, state):
        self.finished = False
    def deactivateAction(self):
        self.finished = True
    def findDirectionToFaceObject(self, playerObjX, playerObjY, playerAngle, targetObjX, targetObjY):
        offsetVector = (targetObjX - playerObjX, targetObjY - playerObjY)

        objectDeg = math.degrees(math.atan2(offsetVector[1], offsetVector[0])) % 360

        playerAngle = (playerAngle + 90) % 360
        objectDeg = (objectDeg + 90) % 360

        angleDirection = objectDeg - playerAngle
        if(abs(angleDirection) < 360 - abs(angleDirection)):
            angleMagnitude = abs(angleDirection)
            if angleDirection > 0:
                print(f"moving {angleMagnitude} and turning left")
                return (angleMagnitude, np.copy(actionMapping["turn_left"]))
            print(f"moving {angleMagnitude} and turning right")
            return (angleMagnitude, np.copy(actionMapping["turn_right"]))
        else:
            angleMagnitude = 360 - abs(angleDirection)
            if angleDirection > 0:
                print(f"angle is {angleMagnitude} and turning right")
                return (angleMagnitude, np.copy(actionMapping["turn_right"]))
            print(f"angle is {angleMagnitude} and turning left")
            return (angleMagnitude, np.copy(actionMapping["turn_left"]))
        
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

        
        leftRightMovementFactor = actionMapping["move_left"] if np.array_equal(tempAngleDirection, actionMapping["turn_left"]) else actionMapping["move_right"]
        print(f"idea of moving left is: {np.array_equal(tempAngleDirection, actionMapping["turn_left"])}")
        if(abs(tempAngleVal) < 1):
            trueActionVector += actionMapping["move_forward"]
            print("moving forward too")
        elif(abs(tempAngleVal) < 89):
            print("moving forward too along with left/right")
            trueActionVector += leftRightMovementFactor + actionMapping["move_forward"]
        elif(abs(tempAngleVal) < 91):
            print("moving solely left/right")
            trueActionVector += leftRightMovementFactor
        elif(abs(tempAngleVal) < 179):
            trueActionVector += leftRightMovementFactor + actionMapping["move_backward"]
            print("moving backward along with left/right")
        else:
            trueActionVector += actionMapping["move_backward"]
            print("moving backward too")

        print(trueActionVector)

        return trueActionVector, trueAngleVal

class SwitchWeapon(RealActions):
    def activateAction(self, state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        self.weaponToSwitchTo = "chaingun" if currentWeapon == 3 else "shotgun"
        previousAction += actionMapping[f"select_{self.weaponToSwitchTo}"]
        super().activateAction(state)
        
    def updateTickAndReturnAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables        
        if((self.weaponToSwitchTo == "chaingun" and currentWeapon == 4) or (self.weaponToSwitchTo == "shotgun" and currentWeapon == 3)):
            self.deactivateAction()

        return previousAction, 0
                
TICKS_FIREANDSTRAFE_IS_ACTIVE = 100

class FireAndStrafe(RealActions):
    def activateAction(self, state):
        self.strafeDirection = 90 if random.random() > 0.5 else 270
        self.currentTick = 0
        super().activateAction(state)
    def updateTickAndReturnAction(self, state):        
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        count, targetX, targetY = getClosestObjPosition(posX, posY, state, "NearestEnemy")
        if(count == 0): 
            self.deactivateAction()
            return previousAction, -10
        
        chosenAction, angle = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, self.strafeDirection)

        
        if(angle < 1):
            chosenAction += actionMapping["attack"]

        self.currentTick += 1
        if(self.currentTick >= TICKS_FIREANDSTRAFE_IS_ACTIVE):
            self.deactivateAction()

        previousAction = chosenAction
        return chosenAction, 0

TICKS_DIRECTLYFLEE_IS_ACTIVE = 250

class DirectlyFlee(RealActions):
    def activateAction(self, state):
        self.strafeDirection = 135 if random.random() > 0.5 else 225
        self.currentTick = 0
        super().activateAction(state)
    def updateTickAndReturnAction(self, state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        count, targetX, targetY = getClosestObjPosition(posX, posY, state, "NearestEnemy")
        if(count == 0):
            self.deactivateAction()
            return previousAction, -10
        
        if(abs(posX - xBounds[0]) < 50 or abs(posX - xBounds[1]) < 50 or abs(posY - yBounds[0]) or abs(posY - xBounds[1]) < 50):
            self.deactivateAction()
            return previousAction, 0
        
        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, self.strafeDirection)

        print(f"current position is {posX}, {posY}, taret position is {targetX, targetY}, current angle is {angle}")
        
        self.currentTick += 1
        if(self.currentTick >= TICKS_DIRECTLYFLEE_IS_ACTIVE):
            self.deactivateAction()

        previousAction = chosenAction
        return chosenAction, 0
        
class GoToHealth(RealActions):
    def activateAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        self.prevHealth = health
        self.maxHealth = 100
        super().activateAction(state)
    def updateTickAndReturnAction(self,state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

        if(health >= self.maxHealth):
            if(self.prevHealth < health):
                # ok good, so we won
                self.deactivateAction()
                return previousAction, 0
            else:
                # do punishment here!
                self.deactivateAction()
                return previousAction, -30
        


        count, targetX, targetY = getClosestObjPosition(posX, posY, state, "Medikit")
        if(count == 0):
            self.deactivateAction()
            # DO SOME PUNISHMENT!
            return previousAction, -30
        
        if(health > self.prevHealth):
            self.deactivateAction()
            return previousAction, 0
        
        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, 0)
        
        previousAction = chosenAction
        self.prevHealth = health
        return chosenAction, 0
class GoToAmmo(RealActions):
    def activateAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        self.ammoTypeToGet = "shells" if currentWeapon == 3 else "clips"
        self.prevAmmo = firstWepAmmo if currentWeapon == 3 else secondWepAmmo
        self.maxAmmo = 50 if currentWeapon == 3 else 200
        super().activateAction(state)
    def updateTickAndReturnAction(self,state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

        currentAmmo = firstWepAmmo if currentWeapon == 3 else secondWepAmmo

        if(currentAmmo == self.maxAmmo):
            if(self.prevAmmo < currentAmmo):
                # ok good, so we won
                self.deactivateAction()
                return previousAction, 15
            else:
                # do punishment here!
                self.deactivateAction()
                return previousAction, -15
        


        count, targetX, targetY = getClosestObjPosition(posX, posY, state, self.ammoTypeToGet)

        print(f"current position is {posX}, {posY}, taret position is {targetX, targetY}, current angle is {angle}")
        if(count == 0):
            self.deactivateAction()
            # DO SOME PUNISHMENT!
            return previousAction, -20
        
        if(currentAmmo > self.prevAmmo):
            self.deactivateAction()
            return previousAction, 15
        
        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, 0)
        
        previousAction = chosenAction
        self.prevAmmo = currentAmmo
        return chosenAction, 0
    
class GoToArmor(RealActions):
    def activateAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        self.prevArmor = armor
        self.maxArmor = 200
        super().activateAction(state)
    def updateTickAndReturnAction(self,state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

        if(armor == self.maxArmor):
            if(self.prevArmor < armor):
                # ok good, so we won
                self.deactivateAction()
                return previousAction, 0
            else:
                # do punishment here!
                self.deactivateAction()
                return previousAction, -20
        


        count, targetX, targetY = getClosestObjPosition(posX, posY, state, "armor")

        print(f"current position is {posX}, {posY}, taret position is {targetX, targetY}, current angle is {angle}")

        if(count == 0):
            self.deactivateAction()
            # DO SOME PUNISHMENT!
            return previousAction, -20
        
        if(armor > self.prevArmor):
            self.deactivateAction()
            return previousAction, 0
        
        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, 0)
        
        previousAction = chosenAction
        self.prevArmor = armor
        return chosenAction, 0
    
class MoveRandom(RealActions):
    def activateAction(self, state):
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        self.targetPoint = (random.random() * (xBounds[1] - xBounds[0]) + xBounds[0], random.random() * (yBounds[1] - yBounds[0]) + yBounds[0])
        super().activateAction(state)
    def updateTickAndReturnAction(self, state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

        if ((((posY - self.targetPoint[1]) ** 2 + (posX - self.targetPoint[0]) ** 2) ** (1/2)) < distanceUntilTooClose / 2):
            self.deactivateAction()
            return previousAction, 0
        
        print(f"current position is {posX}, {posY}, taret position is {self.targetPoint[0], self.targetPoint[1]}, current angle is {angle}")

        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, self.targetPoint[0], self.targetPoint[1], 0)

        previousAction = chosenAction
        return chosenAction, 9
    
def computeEnemyCentroid(state):
    positionArrays = []
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           positionArrays.append((object.position_x, object.position_y))
    
    if(len(positionArrays) > 0):
        return (True, sum(enemy[0] for enemy in positionArrays) / len(positionArrays), sum(enemy[1] for enemy in positionArrays) / len(positionArrays))
    return (False, None, None)

def findBestRunwaySpot(state):
    success, centerX, centerY = computeEnemyCentroid(state)
    if not success:
        return None

    possibleXVals = np.linspace(xBounds[0], xBounds[1], 3)
    possibleYVals = np.linspace(yBounds[0], yBounds[1], 3)

    bestTarget = (0, 0)
    bestDistance = 100000000

    for x in possibleXVals:
        for y in possibleYVals:
            currentDist = ((centerY - y) ** 2 + (centerX - x) ** 2) ** (1/2)
            if (currentDist < bestDistance):
                bestTarget = (x, y)
                bestDistance = currentDist

    return bestTarget

class RunAway(RealActions):
    def activateAction(self, state):
        self.target = findBestRunwaySpot(state)
        super().activateAction(state)
    def updateTickAndReturnAction(self, state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

        if(self.target == None):
            # punish first
            self.deactivateAction()
            return previousAction, -25

        if ((((posY - self.target[1]) ** 2 + (posX - self.target[0]) ** 2) ** (1/2)) < distanceUntilTooClose * 2):
            self.deactivateAction()
            return previousAction, 0
        
        print(f"current position is {posX}, {posY}, taret position is {self.target[0], self.target[1]}, current angle is {angle}")

        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, self.target[0], self.target[1], 0)

        previousAction = chosenAction
        return chosenAction, 0

class ChargeIn(RealActions):
    def activateAction(self, state):
        super().activateAction(state)
    def updateTickAndReturnAction(self, state):
        global previousHealth, lastTickHurtCalled, previousAction
        health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
        hasEnemies, targetX, targetY = computeEnemyCentroid(state)

        if(not hasEnemies):
            # punish here
            self.deactivateAction()
            return previousAction, -20

        if ((((posY - targetY) ** 2 + (posX - targetX) ** 2) ** (1/2)) < distanceUntilTooClose * 3):
            self.deactivateAction()
            return previousAction, 0
        
        print(f"current position is {posX}, {posY}, taret position is {targetX, targetY}, current angle is {angle}")

        chosenAction, _ = self.findMovementToMoveRelativeToObject(posX, posY, angle, targetX, targetY, 0)

        previousAction = chosenAction
        return chosenAction, 0