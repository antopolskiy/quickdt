package quickdt.predictiveModelOptimizer;
/**
 * Created by alexanderhawk on 3/4/14.
 */
public class ParameterToOptimize {
    public Properties properties;
    public ValueWithPreviousValue trialValues;
    public ValueWithPreviousValue trialErrors;

    public ParameterToOptimize(Properties properties){
        this.properties = properties;
        trialValues = new ValueWithPreviousValue();
        trialErrors = new ValueWithPreviousValue();
    }

    public ParameterToOptimize(ParameterToOptimize parameter) {
        this.properties = new Properties(parameter.properties);
        this.trialErrors = new ValueWithPreviousValue(parameter.trialErrors);
        this.trialValues = new ValueWithPreviousValue(parameter.trialValues);
    }


    class ValueWithPreviousValue {
        public Object current;
        public Object previous;
        
        public ValueWithPreviousValue(){}

        public ValueWithPreviousValue(ValueWithPreviousValue valueWithPreviousValue)  {
            this.current = valueWithPreviousValue.current;
            this.previous = valueWithPreviousValue.previous;
        }

        public void setPrevious() {
            previous = current;
        }
    }
}
