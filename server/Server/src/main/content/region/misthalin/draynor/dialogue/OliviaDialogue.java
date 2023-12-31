package content.region.misthalin.draynor.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Handles the OliviaDialogue dialogue.
 * @author 'Vexia
 */
@Initializable
public class OliviaDialogue extends DialoguePlugin {

	public OliviaDialogue() {

	}

	public OliviaDialogue(Player player) {
		super(player);
	}

	@Override
	public int[] getIds() {
		return new int[] { 2233, 2572 };
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "Yes", "No", "Where do I get rarer seeds from?");
			stage = 1;
			break;
		case 1:
			switch (buttonId) {
			case 1:
				end();
				npc.openShop(player);
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.NEUTRAL, "No, thanks.");
				stage = 20;
				break;
			case 3:
				interpreter.sendDialogues(player, FacialExpression.ASKING, "Where do I get rarer seeds from?");
				stage = 40;
				break;

			}
			break;
		case 20:
			end();
			break;
		case 40:
			interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "The Master Farmers usually carry a few rare seeds", "around with them, although I don't know if they'd want", "to part with them for any price to be honest.");
			stage = 41;
			break;
		case 41:
			end();
			break;
		}
		return true;
	}

	@Override
	public DialoguePlugin newInstance(Player player) {

		return new OliviaDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Would you like to trade in seeds?");
		stage = 0;
		return true;
	}
}
