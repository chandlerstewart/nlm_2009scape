package content.region.misc.miscellania.dialogue

import core.game.dialogue.DialoguePlugin
import core.game.dialogue.FacialExpression
import core.game.node.entity.npc.NPC
import core.game.node.entity.player.Player
import core.plugin.Initializable
import org.rs09.consts.NPCs

/**
 * @author qmqz
 */

@Initializable
class FullangrDialogue(player: Player? = null) : core.game.dialogue.DialoguePlugin(player){

    override fun open(vararg args: Any?): Boolean {
        npc = args[0] as NPC
        npc(core.game.dialogue.FacialExpression.OLD_DEFAULT,"Good day, sir.").also { stage = 0 }
        return true
    }

    override fun handle(interfaceId: Int, buttonId: Int): Boolean {
        when(stage){
            0 -> {
                options("What are you doing down here?", "Good day.").also { stage++ }
            }

            1 -> when(buttonId){
                1 -> {
                    player(core.game.dialogue.FacialExpression.FRIENDLY, "What are you doing down here?").also { stage = 10 }
                }

                2 -> {
                    player(core.game.dialogue.FacialExpression.NEUTRAL, "Good day.").also { stage = 99 }
                }
            }

            10 -> {
                npc(core.game.dialogue.FacialExpression.OLD_DEFAULT,"I'm working on the digging, of course.",
                    "It's a small excavation, so only two of us ",
                    "can work on it at a time.").also { stage = 99 }
            }

            99 -> end()
        }
        return true
    }

    override fun newInstance(player: Player?): core.game.dialogue.DialoguePlugin {
        return FullangrDialogue(player)
    }

    override fun getIds(): IntArray {
        return intArrayOf(NPCs.FULLANGR_3934)
    }
}