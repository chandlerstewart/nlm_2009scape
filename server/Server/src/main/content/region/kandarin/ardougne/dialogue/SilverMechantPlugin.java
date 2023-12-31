package content.region.kandarin.ardougne.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Handles the SilverMechantPlugin dialogue.
 * @author 'Vexia
 */
@Initializable
public class SilverMechantPlugin extends DialoguePlugin {

	public SilverMechantPlugin() {

	}

	public SilverMechantPlugin(Player player) {
		super(player);
	}

	@Override
	public int[] getIds() {
		return new int[] { 569 };
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "Yes please.", "No, thank you.");
			stage = 1;
			break;
		case 1:
			switch (buttonId) {
			case 1:
				end();
				npc.openShop(player);
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.NEUTRAL, "No, thank you.");
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

		return new SilverMechantPlugin(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Silver! Silver! Best prices for buying and selling in all", "Kandarin!");
		stage = 0;
		return true;
	}
}
