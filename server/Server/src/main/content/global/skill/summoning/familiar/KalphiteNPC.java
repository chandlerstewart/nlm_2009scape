package content.global.skill.summoning.familiar;

import core.plugin.Initializable;
import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.Metamorphosis;
import core.game.node.entity.npc.NPC;
import core.game.node.entity.player.Player;

/**
 * Handles the kalphite princess pet.
 * @author Empathy
 *
 */
@Initializable
public class KalphiteNPC extends Metamorphosis {

	/**
	 * The kalphite ids.
	 */
	private static final int[] KALPHITE_IDS = new int[] { 8602, 8603 };

	/**
	 * Constructs a new {@code KalphiteNPC} object.
	 */
	public KalphiteNPC() {
		super(KALPHITE_IDS);
	}


	@Override
	public DialoguePlugin getDialoguePlugin() {
		return new KalphitePrincessDialogue();
	}
	
	/**
	 * Handles the KalphitePrincess dialogue.
	 * @author Empathy
	 *
	 */
	public final class KalphitePrincessDialogue extends DialoguePlugin {

		/**
		 * Constructs a new {@code KalphitePrincessDialogue} {@code Object}.
		 */
		public KalphitePrincessDialogue() {
			/**
			 * empty.
			 */
		}

		/**
		 * Constructs a new {@code KalphitePrincessDialogue} {@code Object}.
		 * 
		 * @param player the player.
		 */
		public KalphitePrincessDialogue(Player player) {
			super(player);
		}

		@Override
		public DialoguePlugin newInstance(Player player) {
			return new KalphitePrincessDialogue(player);
		}

		@Override
		public boolean open(Object... args) {
			npc = (NPC) args[0];
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "What is it with your kind and potato cactus?");
			stage = 0;
			return true;
		}

		@Override
		public boolean handle(int interfaceId, int buttonId) {
			switch (stage) {
			case 0:
				interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Truthfully?");
				stage = 1;
				break;
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Yeah, please.");
				stage = 2;
				break;
			case 2:
				interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Soup. We make a fine soup with it.");
				stage = 3;
				break;
			case 3:
				interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Kalphites can cook?");
				stage = 4;
				break;
			case 4:
				interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Nah, we just collect it and put it there because we", "know fools like yourself will come down looking for it", "then inevitably be killed by my mother.");
				stage = 5;
				break;
			case 5:
				interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Evidently not, that's how I got you!");
				stage = 6;
				break;
			case 6:
				interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Touch�.");
				stage = 7;
				break;
			case 7:
				end();
				break;
			}
			return true;
		}

		@Override
		public int[] getIds() {
			return new int[] { 8602, 8603 };
		}
	}
}
