package content.global.ame.events.treespirit

import core.game.node.entity.Entity
import core.game.node.entity.npc.NPC
import content.global.ame.RandomEventNPC
import core.api.utils.WeightBasedTable
import kotlin.math.max


val ids = 438..443

class TreeSpiritRENPC(override var loot: WeightBasedTable? = null) : RandomEventNPC(438){
    override fun talkTo(npc: NPC) {}
    override fun init() {
        super.init()
        val index = max(0,(player.properties.combatLevel / 20) - 1)
        val id = ids.toList()[index]
        this.transform(id)
        this.attack(player)
        sendChat("Leave these woods and never return!")
        this.isRespawn = false
    }
    override fun finalizeDeath(killer: Entity?) {
        super.finalizeDeath(killer)
    }
    override fun tick() {
        if(!player.location.withinDistance(this.location,8)){
            this.terminate()
        }
        super.tick()
        if(!player.viewport.currentPlane.npcs.contains(this)) this.clear()
    }
}