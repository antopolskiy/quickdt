package quickdt;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public final class OrdinalBranch extends Branch {
    private static final  Logger logger =  LoggerFactory.getLogger(OrdinalBranch.class);

	private static final long serialVersionUID = 4456176008067679801L;
	public final float threshold;

	public OrdinalBranch(Node parent, final String attribute, final float threshold) {
		super(parent, attribute);
		this.threshold = threshold;

	}

	@Override
	protected boolean decide(final Attributes attributes) {
        final Serializable value = attributes.get(attribute);
        if (!(value instanceof Number)) {
            throw new RuntimeException("Expecting a number as the value of "+attribute+" but got "+value +" of type "+value.getClass().getSimpleName());
        }
        final float valueAsFloat = ((Number) value).floatValue();
		return valueAsFloat > threshold;
	}

	@Override
	public String toString() {
		return attribute + " > " + threshold;
	}

	@Override
	public String toNotString() {
		return attribute + " <= " + threshold;

	}
}
