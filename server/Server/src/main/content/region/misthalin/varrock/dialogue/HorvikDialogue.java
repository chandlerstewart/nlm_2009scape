package content.region.misthalin.varrock.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Represents the horvik dialogue plugin.
 * @author 'Vexia
 * @version 1.0
 */
@Initializable
public final class HorvikDialogue extends DialoguePlugin {

	/**
	 * Constructs a new {@code HorvikDialogue} {@code Object}.
	 */
	public HorvikDialogue() {
		/**
		 * empty.
		 */
	}

	/**
	 * Constructs a new {@code HorvikDialogue} {@code Object}.
	 * @param player the player.
	 */
	public HorvikDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new HorvikDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Hello, do you need any help?");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "No, thanks. I'm just looking around.", "Do you want to trade?");
			stage = 1;
			break;
		case 1:

			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "No, thanks. I'm just looking around.");
				stage = 10;
				break;
			case 2:
				end();
				npc.openShop(player);
				break;
			}

			break;
		case 10:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Well, come and see me if you're ever in need of armour!");
			stage = 11;
			break;
		case 11:
			end();
			break;
		}

		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 549 };
	}
}
