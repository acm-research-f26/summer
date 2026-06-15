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
            
class FireAndStrafe(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class DirectlyFlee(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class GoToHealth(RealActions):
    def updateTickAndReturnAction(self,state):
        pass

class GoToAmmo(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class GoToArmor(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class MoveRandom(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class RunAway(RealActions):
    def updateTickAndReturnAction(self, state):
        pass

class ChargeIn(RealActions):
    def updateTickAndReturnAction(self, state):
        pass