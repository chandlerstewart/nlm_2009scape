package content.region.kandarin.ardougne.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Handles the SpiceSellerPlugin dialogue.
 * @author 'Vexia
 */
@Initializable
public class SpiceSellerPlugin extends DialoguePlugin {

	public SpiceSellerPlugin() {

	}

	public SpiceSellerPlugin(Player player) {
		super(player);
	}

	@Override
	public int[] getIds() {
		return new int[] { 572 };
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "Yes.", "No.");
			stage = 1;
			break;
		case 1:
			switch (buttonId) {
			case 1:
				end();
				npc.openShop(player);
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "No, thanks.");
				stage = 20;
				break;

			}
			break;
		case 20:
			end();
			break;
		}
		return true;
	}

	@Override
	public DialoguePlugin newInstance(Player player) {

		return new SpiceSellerPlugin(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Are you interested in buying or selling spice?");
		stage = 0;
		return true;
	}
}
