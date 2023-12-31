package content.region.morytania.canifis.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.plugin.Initializable;
import content.global.skill.crafting.TanningProduct;
import core.game.node.entity.npc.NPC;
import core.game.node.entity.player.Player;
import core.game.node.item.Item;

/**
 * Handles the SbottDialogue dialogue.
 * @author 'Vexia
 */
@Initializable
public class SbottDialogue extends DialoguePlugin {

	public SbottDialogue() {

	}

	public SbottDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {

		return new SbottDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Hello stranger. Would you like to me to tan any hides for", "you?");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			// interpreter.sendDialogues(npc, FacialExpression.NORMAL,
			// "Soft leather - 2 gp per hide","Hard leather - 5 gp per hide","Snakeskins - 25 gp per hide","Dragon leather - 45 gp per hide.");
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Soft leather - 1 gp per hide", "Hard leather - 3 gp per hide", "Snakeskins - 20 gp per hide", "Dragon leather - 20 gp per hide.");
			stage = 1;
			break;
		case 1:
			player.getInventory().refresh();
			Item items[] = player.getInventory().toArray();
			for (int i = 0; i < items.length; i++) {
				if (items[i] == null) {
					continue;
				}
				if (TanningProduct.forItemId(items[i].getId()) != null) {
					interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "I see you have brought me some hides.", "Would you like me to tan them for you?");
					stage = 100;
					return true;
				}
			}
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "No thanks, I haven't any hides.");
			stage = 2;
			break;
		case 2:
			end();
			break;
		case 100:
			interpreter.sendOptions("Select an Option", "Yes please.", "No thanks.");
			stage = 101;
			break;
		case 101:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Yes please.");
				stage = 210;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.NEUTRAL, "No thanks.");
				stage = 200;
				break;
			}
			break;
		case 210:
			end();
			TanningProduct.open(player, 2824);
			break;
		case 200:
			interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "Very well, sir, as you wish.");
			stage = 201;
			break;
		case 201:
			end();
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 1041 };
	}
}
